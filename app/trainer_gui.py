import os, sys, json, threading, queue, shutil, glob, time, subprocess, gc
from pathlib import Path

# Configuración para reducir fragmentación de VRAM y permitir segmentos expandibles
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ====== Conversor directo Label Studio JSON -> YOLO-Seg ======

def build_yolo_seg_from_labelstudio(json_path: str, images_dir: str, out_dir: str, class_names: list,
                                    allowed_labels: list | set | None = None,
                                    allowed_geoms: list | set | None = None,
                                    val_split: float = 0.2,
                                    seed: int = 42,
                                    negatives_dir: str | None = None):
    """
    Convierte un export JSON/NDJSON de Label Studio con anotaciones de segmentación a
    un dataset YOLO-Seg estándar, copiando las imágenes y creando un .txt por imagen
    con los polígonos normalizados. Soporta:
      - brush / brushlabels (RLE) -> contornos a polígonos
      - polygonlabels (lista de puntos)
      - rectanglelabels (se convierte a polígono de 4 vértices, con rotación si aplica)

    Estructura salida:
      out_dir/
        images/train/*.jpg|png, images/val/*.jpg|png
        labels/train/*.txt, labels/val/*.txt
        data.yaml
    """
    from label_studio_converter import brush as brush_utils
    import numpy as np
    import cv2
    import random

    out = Path(out_dir)
    images_train = out / "images" / "train"
    labels_train = out / "labels" / "train"
    images_val = out / "images" / "val"
    labels_val = out / "labels" / "val"
    
    for d in [images_train, labels_train, images_val, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Lee JSON o NDJSON
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    # Lista de muestras válidas: [(src_img_path, content_lines, label_stem)]
    valid_samples = []

    images_dir_p = Path(images_dir) if images_dir else None

    def find_image_local(img_rel: str):
        if not images_dir_p: return None
        # Normalización básica
        img_rel = img_rel.replace("\\", "/")
        
        # 1. Busqueda directa combinada
        cand = images_dir_p / img_rel
        if cand.exists(): return cand
        
        # 2. Por nombre base
        name = Path(img_rel).name
        stem = Path(img_rel).stem
        
        # Primero busqueda exacta del nombre en todo el árbol (puede ser lento si hay muchas fotos)
        # Optimizacion: buscar primero en root, luego rglob
        if (images_dir_p / name).exists():
            return images_dir_p / name
            
        for p in images_dir_p.rglob(name): return p
        
        # 3. Por stem con extensiones soportadas
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
        for p in images_dir_p.rglob(f"*{stem}*"):
            if p.suffix.lower() in exts: return p
            
        # 4. Fallback agresivo: buscar cualquier cosa que contenga el stem
        for ext in exts:
            for p in images_dir_p.rglob(f"*{stem}{ext}"): return p
            
        return None

    def _clip01(arr):
        return np.clip(arr, 0.0, 1.0)

    # --- BLOCK 1: COCO Format ---
    if isinstance(data, dict) and ('images' in data) and ('annotations' in data):
        images = data.get('images', []) or []
        annos = data.get('annotations', []) or []
        cats = data.get('categories', []) or []
        
        allowed_labels_set = set([str(x) for x in allowed_labels]) if allowed_labels else None
        allowed_labels_norm = set([str(x).strip().lower() for x in allowed_labels_set]) if allowed_labels_set else None
        allowed_geoms_set = set(allowed_geoms) if allowed_geoms else {"polygon"}

        # Helpers helpers for COCO
        name_to_id = {}
        cat_id_to_name = {}
        target_class_names = list(class_names) if class_names else []
        for c in cats:
            nm = str(c.get('name', '')).strip()
            cid = int(c.get('id')) if isinstance(c.get('id'), (int, float)) else None
            if nm:
                cat_id_to_name[cid] = nm
                nm_norm = nm.lower()
                if (allowed_labels_norm is None) or (nm_norm in allowed_labels_norm):
                    if nm not in target_class_names:
                         target_class_names.append(nm)
                    name_to_id[nm] = target_class_names.index(nm)
        
        # update global class names reference
        class_names[:] = target_class_names 

        def _get_cls_by_catid(cid: int) -> int:
            nm = cat_id_to_name.get(cid)
            if not nm: return -1
            nm_norm = nm.strip().lower()
            if (allowed_labels_norm is not None) and (nm_norm not in allowed_labels_norm): return -1
            if nm not in name_to_id:
                name_to_id[nm] = len(class_names)
                class_names.append(nm)
            return name_to_id[nm]

        id_to_img_info = {int(im.get('id')): im for im in images}
        
        # First pass: group annotations by image_id
        annos_by_img = {}
        for a in annos:
            iid = a.get('image_id')
            if iid is not None:
                annos_by_img.setdefault(int(iid), []).append(a)

        for img_id, img_annos in annos_by_img.items():
            info = id_to_img_info.get(img_id)
            if not info: continue
            
            # Find file
            fname = str(info.get('file_name') or info.get('path') or '')
            # A veces viene codificado o con paths relativos largos
            fname = os.path.basename(fname.split('?')[0])
            
            src_img = find_image_local(fname)
            if not src_img:
                continue
            
            W = float(max(1, info.get('width', 1)))
            H = float(max(1, info.get('height', 1)))
            
            lines = []
            for a in img_annos:
                cls = _get_cls_by_catid(int(a.get('category_id'))) if a.get('category_id') is not None else -1
                if cls < 0: continue
                
                seg = a.get('segmentation')
                made = False
                if ('polygon' in allowed_geoms_set) and isinstance(seg, list) and seg:
                    for poly in seg:
                         if isinstance(poly, list) and len(poly) >= 6 and (len(poly) % 2 == 0):
                            xs = np.array(poly[0::2], dtype=np.float32) / W
                            ys = np.array(poly[1::2], dtype=np.float32) / H
                            xs = _clip01(xs)
                            ys = _clip01(ys)
                            if xs.size >= 3:
                                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
                                lines.append(f"{cls} {coords}")
                                made = True
                
                if (not made) and ('polygon' in allowed_geoms_set):
                    bbox = a.get('bbox')
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x, y, w, h = [float(v) for v in bbox]
                        rect = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)
                        rect[:,0] /= W
                        rect[:,1] /= H
                        rect[:,0] = _clip01(rect[:,0])
                        rect[:,1] = _clip01(rect[:,1])
                        coords = " ".join(f"{xv:.6f} {yv:.6f}" for xv, yv in rect)
                        lines.append(f"{cls} {coords}")

            if lines:
                valid_samples.append((src_img, list(set(lines)), src_img.stem))

    # --- BLOCK 2: Label Studio Format ---
    else:
        tasks = data if isinstance(data, list) else [data]
        
        # Setup classes
        target_class_names = list(class_names) if class_names else []
        allowed_labels_set = set([str(x) for x in allowed_labels]) if allowed_labels else None
        allowed_labels_norm = set([str(x).strip().lower() for x in allowed_labels_set]) if allowed_labels_set else None
        if allowed_geoms:
            allowed_geoms_set = set(allowed_geoms)
        else:
            allowed_geoms_set = {"brush", "polygon"}
            
        name_to_id = {n: i for i, n in enumerate(target_class_names)}
        
        def _add_class(label: str) -> int:
            lab_norm = label.strip().lower()
            if allowed_labels_norm is not None and lab_norm not in allowed_labels_norm:
                return -1
            if label not in name_to_id:
                name_to_id[label] = len(target_class_names)
                target_class_names.append(label)
            return name_to_id[label]
        
        # update global reference
        class_names[:] = target_class_names
        
        for t in tasks:
            image_field = t.get("data", {}).get("image") or t.get("data", {}).get("img") or ""
            img_rel = image_field.split("/")[-1]
            if not img_rel:
                # Caso extremo: task sin imagen
                continue
                
            src_img = find_image_local(img_rel)
            if not src_img:
                continue
            
            lines = []
            anns = []
            anns.extend(t.get("annotations", []) or [])
            anns.extend(t.get("predictions", []) or [])
            
            for ann in anns:
                results_list = ann.get("result", []) or []
                labels_by_id = {}
                for r in results_list:
                    rt = r.get("type") or r.get("from_name")
                    v = r.get("value", {}) or {}
                    if rt in ("labels", "brushlabels", "polygonlabels", "rectanglelabels"):
                        labs = v.get("labels") or v.get("brushlabels") or v.get("polygonlabels") or v.get("rectanglelabels")
                        if labs:
                            labels_by_id[r.get("id")] = labs
                
                for res in results_list:
                    rtype = res.get("type") or res.get("from_name")
                    val = res.get("value", {}) or {}
                    
                    # Brush
                    if (rtype in ("brush", "brushlabels") or "rle" in val) and "rle" in val:
                        if "brush" not in allowed_geoms_set: continue
                        h = val.get("original_height") or val.get("height") or res.get("original_height") or res.get("height")
                        w = val.get("original_width") or val.get("width") or res.get("original_width") or res.get("width")
                        rle = val.get("rle")
                        if rle is None or not (h and w): continue
                        
                        labels_v = val.get("labels") or val.get("brushlabels") or labels_by_id.get(res.get("id"))
                        if not labels_v: continue
                        label = labels_v[0] if isinstance(labels_v, list) and labels_v else str(labels_v)
                        cls = _add_class(label.strip())
                        if cls < 0: continue
                        
                        try:
                            mask_flat = brush_utils.decode_rle(rle)
                            H, W = int(h), int(w)
                            mask = None
                            arr = np.asarray(mask_flat)
                            if arr.size == H*W*4: mask = arr.reshape(H,W,4)[:,:,3]
                            elif arr.size == H*W: mask = arr.reshape(H,W)
                            
                            if mask is not None:
                                bin_mask = (mask > 0).astype("uint8")
                                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for cnt in contours:
                                    if cv2.contourArea(cnt) <= 0.0: continue
                                    peri = cv2.arcLength(cnt, True)
                                    eps = 0.0015 * peri
                                    approx = cv2.approxPolyDP(cnt, eps, True)
                                    if len(approx) < 3:
                                        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
                                        approx = box.reshape(-1, 1, 2)
                                    pts = approx.reshape(-1, 2).astype(np.float32)
                                    xs = _clip01(pts[:,0] / float(w))
                                    ys = _clip01(pts[:,1] / float(h))
                                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
                                    lines.append(f"{cls} {coords}")
                        except Exception:
                            pass

                    # Polygon
                    elif (rtype in ("polygonlabels", "polygon")) or ("points" in val):
                        if "polygon" not in allowed_geoms_set: continue
                        pts = val.get("points") or []
                        if len(pts) < 3: continue
                        labels_v = val.get("labels") or val.get("polygonlabels") or labels_by_id.get(res.get("id"))
                        if not labels_v: continue
                        label = (labels_v[0] if isinstance(labels_v, list) and labels_v else str(labels_v)).strip()
                        cls = _add_class(label)
                        if cls < 0: continue
                        
                        xs = np.array([float(p[0]) for p in pts], dtype=np.float32)
                        ys = np.array([float(p[1]) for p in pts], dtype=np.float32)
                        
                        # Normalize logic
                        m = float(max(np.max(xs), np.max(ys)))
                        if m > 100.0:
                             h = float(val.get("original_height") or val.get("height") or 1.0)
                             w = float(val.get("original_width") or val.get("width") or 1.0)
                             xs /= max(1.0, w)
                             ys /= max(1.0, h)
                        elif m > 1.0: # 0-100
                             xs /= 100.0
                             ys /= 100.0
                        
                        xs = _clip01(xs)
                        ys = _clip01(ys)
                        if xs.size >= 3:
                             coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
                             lines.append(f"{cls} {coords}")
                    
                    # Rectangle
                    elif (rtype in ("rectanglelabels", "rectangle")) or (all(k in val for k in ("x", "y", "width", "height"))):
                         if "polygon" not in allowed_geoms_set: continue
                         labels_v = val.get("labels") or val.get("rectanglelabels") or labels_by_id.get(res.get("id"))
                         if not labels_v: continue
                         label = (labels_v[0] if isinstance(labels_v, list) and labels_v else str(labels_v)).strip()
                         cls = _add_class(label)
                         if cls < 0: continue
                         
                         x, y = float(val.get("x",0)), float(val.get("y",0))
                         wv, hv = float(val.get("width",0)), float(val.get("height",0))
                         rot = float(val.get("rotation") or 0.0)
                         
                         if max(x,y,wv,hv) > 1.5:
                             H = float(val.get("original_height") or val.get("height") or 1.0)
                             W = float(val.get("original_width") or val.get("width") or 1.0)
                             x/=max(1.0,W); y/=max(1.0,H); wv/=max(1.0,W); hv/=max(1.0,H)
                         else:
                             x/=100.0; y/=100.0; wv/=100.0; hv/=100.0
                             
                         cx, cy = x + wv/2.0, y + hv/2.0
                         rect = np.array([[x,y],[x+wv,y],[x+wv,y+hv],[x,y+hv]], dtype=np.float32)
                         
                         if abs(rot) > 1e-3:
                             ang = np.deg2rad(rot)
                             R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
                             rect = (rect - [cx,cy]) @ R.T + [cx,cy]
                         
                         rect[:,0] = _clip01(rect[:,0])
                         rect[:,1] = _clip01(rect[:,1])
                         coords = " ".join(f"{xv:.6f} {yv:.6f}" for xv, yv in rect)
                         lines.append(f"{cls} {coords}")

            if lines:
                # Deduplicate lines
                uniq_lines = list(set(lines))
                valid_samples.append((src_img, uniq_lines, src_img.stem))

    # --- Processing: Split, Write, Negatives ---
    rng = random.Random(seed)
    rng.shuffle(valid_samples)
    
    n_val = int(len(valid_samples) * val_split)
    val_set = valid_samples[:n_val]
    train_set = valid_samples[n_val:]
    
    def write_dataset(subset, is_val=False):
        d_img = images_val if is_val else images_train
        d_lbl = labels_val if is_val else labels_train
        c = 0
        for (src, lines, stem) in subset:
             dst = d_img / src.name
             if not dst.exists():
                 shutil.copy2(src, dst)
             # Use the stem from the src image, not the one from label logic if possible, but consistent
             (d_lbl / f"{dst.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
             c += 1
        return c

    print(f"Writing {len(train_set)} train images, {len(val_set)} val images.")
    write_dataset(train_set, False)
    write_dataset(val_set, True)
    
    # Negatives
    if negatives_dir:
        neg_p = Path(negatives_dir)
        if neg_p.exists():
             exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
             all_negs = [p for p in neg_p.rglob("*") if p.is_file() and p.suffix.lower() in exts]
             rng.shuffle(all_negs)
             
             nn_val = int(len(all_negs) * val_split)
             neg_val = all_negs[:nn_val]
             neg_train = all_negs[nn_val:]
             
             def write_negs(subset, is_val=False):
                 d_img = images_val if is_val else images_train
                 d_lbl = labels_val if is_val else labels_train
                 for src in subset:
                     dst = d_img / src.name
                     if not dst.exists():
                         shutil.copy2(src, dst)
                     (d_lbl / f"{dst.stem}.txt").write_text("", encoding="utf-8")
             
             write_negs(neg_train, False)
             write_negs(neg_val, True)
             print(f"Added {len(neg_train)} train negatives, {len(neg_val)} val negatives.")

    yaml_path = out / "data.yaml"
    yaml_text = (
        f"path: {out.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"task: segment\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return str(yaml_path)

# ====== Utilidades de modelo ======

def _resolve_model_name(model_version: str, model_size: str) -> tuple[str, str, str]:
    ver = model_version.lower().strip()
    sz_full = model_size.lower().strip()
    sz = sz_full.split(" ")[0] if " " in sz_full else sz_full
    prefix = "yolov8"
    if ver == "v8":
        prefix = "yolov8"
    elif ver == "v9":
        prefix = "yolov9" 
    elif ver == "v10":
        # YOLOv10 no soporta segmentacion
        prefix = "yolov8" 
    elif ver == "v11":
        prefix = "yolo11"
    elif ver == "v12":
        prefix = "yolo12"
    elif ver == "v26":
        prefix = "yolo26"
    return ver, sz, f"{prefix}{sz}-seg.pt"

# ====== Entrenamiento YOLO-Seg ======

def train_yolo_seg(
    data_yaml: str,
    imgsz: int,
    epochs: int,
    device: str,
    model_version: str,
    model_size: str,
    batch: int | float | str,
    workers: int,
    log_q: queue.Queue,
    run_dir: str | None = None,
    mask_ratio: int = 4,
    overlap_mask: bool = True,
):
    from ultralytics import YOLO
    from pathlib import Path

    # Parsing de versión y tamaño
    ver = model_version.lower().strip()
    sz_full = model_size.lower().strip()
    sz = sz_full.split(" ")[0] if " " in sz_full else sz_full

    # Construcción del nombre del modelo
    prefix = "yolov8"
    if ver == "v8": prefix = "yolov8"
    elif ver == "v9": prefix = "yolov9"
    elif ver == "v10": prefix = "yolov10"
    elif ver == "v11": prefix = "yolo11"
    elif ver == "v12": prefix = "yolo12"
    elif ver == "v26": prefix = "yolo26"
    
    model_name = f"{prefix}{sz}-seg.pt"
    pretrained_weights = None
    use_seg_cfg = False

    try:
        # Batch puede ser int, float (0.70) o -1
        if str(batch) == "-1":
            batch = -1
        elif isinstance(batch, str) and "." in batch:
            batch = float(batch)
        else:
            batch = int(batch)
            if batch != -1:
                batch = max(1, batch)
    except Exception:
        batch = 8 # Fallback seguro

    try:
        workers = int(workers)
    except Exception:
        workers = 0
    workers = max(0, workers)

    # Ruta a la carpeta local de modelos pre-descargados
    try:
        project_root = Path(__file__).resolve().parent.parent
        modelos_root = project_root / "models" / "YoloPresets"
        model_path = modelos_root / model_name
        detect_model_path = modelos_root / f"yolo26{sz}.pt" if ver == "v26" else None
        log_q.put(f"Buscando modelo local: {model_path}")
    except Exception:
        modelos_root = None
        model_path = None
        detect_model_path = None
        log_q.put("No se pudo determinar la ruta local para los modelos. Se intentará descargar.")

    log_q.put(f"Cargando modelo: {model_name} (Versión: {ver}, Tamaño: {sz})")

    try:
        if ver == "v12":
            if not modelos_root:
                log_q.put("ERROR: ruta de modelos no disponible para yolo12-seg.yaml.")
                return None
            yaml_local = modelos_root / "yolo12-seg.yaml"
            scale_yaml = modelos_root / f"yolo12{sz}-seg.yaml"
            if not scale_yaml.exists() and yaml_local.exists():
                try:
                    import shutil
                    shutil.copy2(yaml_local, scale_yaml)
                except Exception as e:
                    log_q.put(f"WARN: no se pudo crear {scale_yaml.name}: {e}")
            
            final_model_path = str(scale_yaml) if scale_yaml.exists() else str(yaml_local)
            use_seg_cfg = True
            log_q.put(f"YOLOv12 (segmentacion) usara config: {final_model_path} (size={sz})")
        elif ver == "v26":
            yaml_local = None
            try:
                import ultralytics
                yaml_local = Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "26" / "yolo26-seg.yaml"
                uv = getattr(ultralytics, "__version__", "")
                if uv:
                    log_q.put(f"Ultralytics: {uv}")
            except Exception:
                yaml_local = None
            if not (yaml_local and yaml_local.exists()):
                log_q.put("ERROR: no se encontro yolo26-seg.yaml. Actualiza ultralytics y reinicia.")
                return None
            final_model_path = str(yaml_local)
            use_seg_cfg = True
            log_q.put(f"YOLOv26 (segmentacion) usara config: {final_model_path}")
            if model_path and model_path.exists():
                pretrained_weights = str(model_path)
            elif detect_model_path and detect_model_path.exists():
                pretrained_weights = str(detect_model_path)
            if pretrained_weights:
                log_q.put(f"YOLOv26: usando pesos locales como pretrained: {pretrained_weights}")
        else:
            final_model_path = str(model_path) if model_path and model_path.exists() else model_name
            if model_path and model_path.exists():
                log_q.put(f"Usando modelo local: {final_model_path}")
            else:
                log_q.put(f"Modelo no encontrado localmente. Ultralytics intentara descargar: {model_name}")

        if use_seg_cfg:
            model = YOLO(final_model_path, task="segment")
        else:
            model = YOLO(final_model_path)

    except Exception as e:
        log_q.put(f"ERROR crítico cargando modelo '{model_name}': {e}")
        log_q.put("Asegúrate de tener la librería ultralytics actualizada (pip install -U ultralytics)")
        return None

    # Callback mínimo para enviar logs a la UI y limpiar VRAM
    def cb(trainer):
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
            
        try:
            loss_val = trainer.label_loss.item() if hasattr(trainer, "label_loss") else None
        except Exception:
            loss_val = None
        log_q.put(
            f"epoch {getattr(trainer, 'epoch', '?') + 1 if hasattr(trainer, 'epoch') else '?'}"
            f"/{getattr(trainer, 'epochs', '?')} | loss={loss_val if loss_val is not None else '...'}"
        )

    try:
        model.add_callback("on_fit_epoch_end", cb)
    except Exception:
        pass

    train_kwargs = dict(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        device=device,
        val=True, # Enable validation since we now have split
        workers=workers,
        batch=batch,
        cache=False,
        mosaic=1.0,
        close_mosaic=15,
        copy_paste=0.3,
        degrees=2.5,
        translate=0.05,
        scale=0.5,
        shear=1.0,
        fliplr=0.5,
        flipud=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.0 if ver != "v26" else None,
        mask_ratio=mask_ratio,
        overlap_mask=overlap_mask,
    )
    if not train_kwargs["dfl"]: train_kwargs.pop("dfl")

    if use_seg_cfg and pretrained_weights:
        train_kwargs["pretrained"] = pretrained_weights
    if run_dir:
        run_path = Path(run_dir)
        if not run_path.name:
            run_path = run_path / time.strftime("run_%Y%m%d_%H%M%S")
        run_path.parent.mkdir(parents=True, exist_ok=True)
        train_kwargs["project"] = str(run_path.parent)
        train_kwargs["name"] = run_path.name

    # Bucle de intentos para recuperar OOM
    while True:
        try:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            log_q.put(f"Memoria CUDA limpiada. Intentando batch={train_kwargs['batch']}, workers={train_kwargs['workers']}")
        except Exception:
            pass

        try:
            # Reinstanciar modelo para evitar grafos corruptos si es reintento
            if use_seg_cfg and final_model_path.endswith(".yaml"):
                model = YOLO(final_model_path, task="segment")
            else:
                model = YOLO(final_model_path)
            
            if use_seg_cfg:
                 # Añadir callback nuevamente al nuevo modelo
                 try: model.add_callback("on_fit_epoch_end", cb)
                 except: pass

            results = model.train(**train_kwargs)
            log_q.put(f"TRAIN_DONE:{results.save_dir}")
            return results.save_dir
            
        except Exception as e:
            e_str = str(e).lower()
            if "out of memory" in e_str or "cuda error" in e_str or "alloc" in e_str:
                log_q.put(f"⚠️ OOM DETECTADO: {e}")
                
                # Politica de reducción
                current_batch = train_kwargs.get("batch", 8)
                
                # Si es AutoBatch (-1), primero probamos un valor fijo seguro
                if current_batch == -1:
                    new_batch = 4
                elif isinstance(current_batch, float):
                    # Si es float (0.7), bajamos a algo fijo
                    new_batch = 2
                else:
                    new_batch = max(1, int(current_batch) // 2)
                
                if new_batch < 1 or (current_batch != -1 and not isinstance(current_batch, float) and new_batch >= int(current_batch)):
                    # Ya no podemos bajar más
                    log_q.put("ERROR CRÍTICO: No se puede reducir más el batch. OOM persiste.")
                    return None
                
                log_q.put(f"♻️ RECUPERANDO: Reduciendo Batch de {current_batch} -> {new_batch}. Reiniciando...")
                train_kwargs["batch"] = new_batch
                train_kwargs["workers"] = 0 # Force 0 workers on retry
                
                # Espera breve
                time.sleep(2)
                continue
            else:
                # Error no relacionado con memoria
                log_q.put(f"ERROR EN ENTRENAMIENTO: {e}")
                return None


def export_trt(ckpt: str, imgsz: int, device: str, log_q: queue.Queue):
    from ultralytics import YOLO
    model = YOLO(ckpt)
    log_q.put("Exportando a TensorRT FP16...")
    model.export(format="engine", imgsz=imgsz, half=True, dynamic=False, device=device)
    log_q.put("EXPORT_DONE")


# ====== UI (Tkinter) ======

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Entrenador YOLOv8-Seg • Tkinter (v0.2)")
        self.geometry("940x720")
        self.log_q = queue.Queue()
        self._make_widgets()
        self.after(100, self._drain_logs)

    def _make_widgets(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # --- Paso 0: Setup
        f0 = ttk.Frame(nb)
        nb.add(f0, text="0) Setup")
        f0.columnconfigure(0, weight=1)
        ttk.Label(
            f0,
            text="Instalacion rapida para preparar dependencias y pesos.",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(8, 4))
        ttk.Button(
            f0,
            text="Instalar/Actualizar dependencias",
            command=self._install_deps_async,
        ).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Button(
            f0,
            text="Descargar modelo seleccionado",
            command=self._download_model_async,
        ).grid(row=2, column=0, sticky="w", padx=6, pady=4)

        # --- Paso 1: Datos
        f1 = ttk.Frame(nb)
        nb.add(f1, text="1) Datos")

        self.var_json = tk.StringVar()
        self.var_imgs = tk.StringVar()
        self.var_negs = tk.StringVar() # NEW
        self.var_yaml = tk.StringVar()
        self.var_classes = tk.StringVar(value="[]")
        self.var_val_split = tk.DoubleVar(value=0.2) # NEW
        self.var_seed = tk.IntVar(value=42) # NEW
        self.var_last_run = tk.StringVar(value="") # Fix for AttributeError

        # Bloque JSON+Imágenes
        row = 0
        frm2 = ttk.LabelFrame(f1, text="Datos de Label Studio (JSON/NDJSON + carpeta de imágenes)")
        frm2.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=10)
        frm2.columnconfigure(0, weight=1)
        ttk.Label(frm2, text="JSON/NDJSON").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(frm2, textvariable=self.var_json, width=70).grid(row=1, column=0, padx=6, sticky="w")
        ttk.Button(frm2, text="Examinar", command=lambda: self._pick_file(self.var_json)).grid(row=1, column=1, padx=4)
        ttk.Button(frm2, text="Cargar etiquetas", command=self._load_labels_from_json).grid(row=1, column=2, padx=4)
        
        ttk.Label(frm2, text="Carpeta de imágenes").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(frm2, textvariable=self.var_imgs, width=70).grid(row=3, column=0, padx=6, sticky="w")
        ttk.Button(frm2, text="Examinar", command=lambda: self._pick_dir(self.var_imgs)).grid(row=3, column=1, padx=4)
        
        # New Negatives row
        ttk.Label(frm2, text="Carpeta Negativos (Opcional)").grid(row=4, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(frm2, textvariable=self.var_negs, width=70).grid(row=5, column=0, padx=6, sticky="w")
        ttk.Button(frm2, text="Examinar", command=lambda: self._pick_dir(self.var_negs)).grid(row=5, column=1, padx=4)

        # Filtros y Split
        row += 1
        filt = ttk.LabelFrame(f1, text="Configuración Dataset")
        filt.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        filt.columnconfigure(0, weight=1)
        
        self.var_geom_brush = tk.BooleanVar(value=False)
        self.var_geom_poly = tk.BooleanVar(value=True)
        self.var_geom_both = tk.BooleanVar(value=False)
        gfrm = ttk.Frame(filt)
        gfrm.grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(gfrm, text="Geometrías:").pack(side="left", padx=(0,4))
        ttk.Checkbutton(gfrm, text="Brush", variable=self.var_geom_brush).pack(side="left", padx=4)
        ttk.Checkbutton(gfrm, text="Polygon", variable=self.var_geom_poly).pack(side="left", padx=4)
        ttk.Checkbutton(gfrm, text="Both", variable=self.var_geom_both).pack(side="left", padx=4)
        
        sfrm = ttk.Frame(filt)
        sfrm.grid(row=0, column=1, sticky="w", padx=20, pady=6)
        ttk.Label(sfrm, text="Val Split:").pack(side="left")
        ttk.Entry(sfrm, textvariable=self.var_val_split, width=5).pack(side="left", padx=4)
        ttk.Label(sfrm, text="Seed:").pack(side="left", padx=(10,0))
        ttk.Entry(sfrm, textvariable=self.var_seed, width=6).pack(side="left", padx=4)

        # Labels checkbox
        ttk.Label(filt, text="Etiquetas (selecciona):").grid(row=1, column=0, sticky="w", padx=6)
        labwrap = ttk.Frame(filt)
        labwrap.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0,6))
        self._labels_canvas = tk.Canvas(labwrap, height=120)
        self._labels_vsb = ttk.Scrollbar(labwrap, orient="vertical", command=self._labels_canvas.yview)
        self._labels_inner = ttk.Frame(self._labels_canvas)
        self._labels_inner.bind("<Configure>", lambda e: self._labels_canvas.configure(scrollregion=self._labels_canvas.bbox("all")))
        self._labels_canvas.create_window((0, 0), window=self._labels_inner, anchor="nw")
        self._labels_canvas.configure(yscrollcommand=self._labels_vsb.set)
        self._labels_canvas.pack(side="left", fill="both", expand=True)
        self._labels_vsb.pack(side="right", fill="y")
        self.lab_chk_vars = {}
        
        row += 1
        ttk.Button(f1, text="Preparar dataset", command=self._prepare_dataset).grid(row=row, column=0, sticky="w", padx=6, pady=8)
        row += 1
        ttk.Label(f1, textvariable=self.var_yaml, foreground="green").grid(row=row, column=0, sticky="w", padx=6)

        # --- Paso 2: Entrenar
        f2 = ttk.Frame(nb)
        nb.add(f2, text="2) Entrenar")
        self.var_imgsz = tk.IntVar(value=1280)
        self.var_epochs = tk.IntVar(value=120)
        self.var_device = tk.StringVar(value="0")
        self.var_version = tk.StringVar(value="v8")
        self.var_size = tk.StringVar(value="s (small)")
        self.var_batch = tk.StringVar(value="8") # Allow text
        self.var_workers = tk.IntVar(value=0)
        self.var_run_dir = tk.StringVar()
        self.var_mask_ratio = tk.IntVar(value=4)
        self.var_overlap_mask = tk.BooleanVar(value=True)

        for i in range(12): f2.columnconfigure(i, weight=0)
        f2.columnconfigure(11, weight=1)

        row = 0; col = 0
        ttk.Label(f2, text="imgsz").grid(row=row, column=col, sticky="e"); col+=1
        ttk.Entry(f2, textvariable=self.var_imgsz, width=6).grid(row=row, column=col, padx=2); col+=1
        
        ttk.Label(f2, text="epochs").grid(row=row, column=col, sticky="e"); col+=1
        ttk.Entry(f2, textvariable=self.var_epochs, width=6).grid(row=row, column=col, padx=2); col+=1
        
        ttk.Label(f2, text="device").grid(row=row, column=col, sticky="e"); col+=1
        ttk.Entry(f2, textvariable=self.var_device, width=4).grid(row=row, column=col, padx=2); col+=1

        ttk.Label(f2, text="Ver.").grid(row=row, column=col, sticky="e"); col+=1
        ver_cb = ttk.Combobox(f2, textvariable=self.var_version, values=["v8", "v9", "v11", "v12", "v26"], state="readonly", width=5)
        ver_cb.grid(row=row, column=col, padx=2); col+=1
        
        ttk.Label(f2, text="Size").grid(row=row, column=col, sticky="e"); col+=1
        size_cb = ttk.Combobox(f2, textvariable=self.var_size, values=["n (nano)", "s (small)", "m (medium)", "l (large)", "x (xlarge)"], width=10, state="readonly")
        size_cb.grid(row=row, column=col, padx=2); col+=1
        size_cb.bind("<<ComboboxSelected>>", self._auto_adjust_params)

        ttk.Button(f2, text="Entrenar", command=self._train_async).grid(row=row, column=col, padx=8); col+=1
        
        row = 1
        ttk.Label(f2, text="batch").grid(row=row, column=0, sticky="e")
        ttk.Entry(f2, textvariable=self.var_batch, width=6).grid(row=row, column=1, padx=2)
        ttk.Label(f2, text="(int/-1/0.7)").grid(row=row, column=2, sticky="w")
        
        ttk.Label(f2, text="workers").grid(row=row, column=3, sticky="e")
        ttk.Entry(f2, textvariable=self.var_workers, width=4).grid(row=row, column=4, padx=2)
        
        # Seg params
        ttk.Label(f2, text="mask_ratio").grid(row=row, column=5, sticky="e")
        ttk.Entry(f2, textvariable=self.var_mask_ratio, width=4).grid(row=row, column=6, padx=2)
        ttk.Checkbutton(f2, text="overlap_mask", variable=self.var_overlap_mask).grid(row=row, column=7, columnspan=2, padx=4)
        ttk.Button(f2, text="Preset Low VRAM", command=self._apply_low_vram).grid(row=row, column=8, padx=4)

        row = 2
        ttk.Label(f2, text="Run Dir").grid(row=row, column=0, sticky="e")
        ttk.Entry(f2, textvariable=self.var_run_dir, width=40).grid(row=row, column=1, columnspan=6, sticky="ew")
        ttk.Button(f2, text="...", command=self._pick_run_dir).grid(row=row, column=7)

        row = 3
        ttk.Label(f2, textvariable=self.var_last_run, foreground="green").grid(row=row, column=0, columnspan=10, sticky="w", padx=10)
        
        advice = (
            "⚠️ AutoBatch (-1) o 0.70 es recomendado para maximizar uso de GPU.\n"
            "Si detect_manchas está corriendo, CIÉRRALO primero."
        )
        ttk.Label(f2, text=advice, foreground="red").grid(row=4, column=0, columnspan=10, sticky="w", padx=10, pady=5)

        # --- Paso 3: Exportar TensorRT
        f3 = ttk.Frame(nb)
        nb.add(f3, text="3) Exportar")
        self.var_ckpt = tk.StringVar()
        ttk.Entry(f3, textvariable=self.var_ckpt, width=80).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(f3, text="best.pt", command=lambda: self._pick_file(self.var_ckpt)).grid(row=0, column=1)
        ttk.Button(f3, text="Exportar", command=self._export_async).grid(row=0, column=2, padx=8)

        # --- Consola
        self.txt = tk.Text(self, height=12)
        self.txt.pack(fill="both", expand=False, padx=6, pady=6)

    def _pick_dir(self, var: tk.StringVar):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def _pick_run_dir(self):
        p = filedialog.askdirectory()
        if p: self.var_run_dir.set(p)

    def _pick_file(self, var: tk.StringVar):
        p = filedialog.askopenfilename(filetypes=[("Todos", "*.*"), ("JSON", "*.json;*.ndjson"), ("PyTorch", "*.pt")])
        if p:
            var.set(p)
            if var is self.var_json:
                try: self._load_labels_from_json()
                except: pass

    def _load_labels_from_json(self):
        path = self.var_json.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("JSON", "Selecciona primero un JSON/NDJSON válido")
            return
        try:
            for w in self._labels_inner.winfo_children(): w.destroy()
        except: pass
        self.lab_chk_vars = {}
        labels_set = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                try: data = json.load(f)
                except:
                    f.seek(0)
                    data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            messagebox.showerror("JSON", f"Error leyendo: {e}")
            return
            
        if isinstance(data, dict) and 'categories' in data:
            for c in data.get('categories', []) or []:
                if c.get('name'): labels_set.add(str(c.get('name')))
        else:
            tasks = data if isinstance(data, list) else [data]
            for t in tasks:
                anns = (t.get("annotations") or []) + (t.get("predictions") or [])
                for ann in anns:
                    for res in ann.get("result", []) or []:
                        v = res.get("value", {}) or {}
                        labs = v.get("labels") or v.get("brushlabels") or v.get("polygonlabels") or v.get("rectanglelabels")
                        if labs:
                           if isinstance(labs, list): 
                               for l in labs: labels_set.add(str(l))
                           else: labels_set.add(str(labs))
                           
        for i, name in enumerate(sorted(labels_set)):
            var = tk.BooleanVar(value=True)
            self.lab_chk_vars[name] = var
            ttk.Checkbutton(self._labels_inner, text=name, variable=var).grid(row=i // 3, column=i % 3, sticky="w", padx=6, pady=2)
        try:
            self.var_classes.set(json.dumps(sorted(labels_set), ensure_ascii=False) if labels_set else "[]")
        except: self.var_classes.set("[]")

    def _prepare_dataset(self):
        try:
            gb = self.var_geom_brush.get()
            gp = self.var_geom_poly.get()
            gboth = self.var_geom_both.get()
            if (gb and gp) and (not gboth):
                messagebox.showerror("Geo", "Activa 'Both' para usar Brush y Polygon a la vez.")
                return
            if (not gb) and (not gp):
                messagebox.showerror("Geo", "Selecciona al menos una geometría.")
                return
            geoms = {"brush", "polygon"} if (gboth or (gb and gp)) else ({"brush"} if gb else {"polygon"})
            
            if not getattr(self, "lab_chk_vars", None):
                messagebox.showerror("Etiquetas", "Carga JSON y etiquetas primero.")
                return
            classes = [n for n, v in self.lab_chk_vars.items() if v.get()]
            if not classes:
                messagebox.showerror("Etiquetas", "Marca al menos una etiqueta.")
                return

            try: project_root = Path(__file__).resolve().parents[3]
            except: project_root = Path.cwd()
            out_dataset = project_root / "datasets" / "yolo_seg"
            if out_dataset.exists():
                try: shutil.rmtree(out_dataset)
                except: pass

            json_path = self.var_json.get().strip()
            imgs_path = self.var_imgs.get().strip()
            if not (json_path and imgs_path):
                messagebox.showerror("Rutas", "Falta JSON o carpeta de imágenes.")
                return
            
            # --- NEW PARAMS ---
            split = self.var_val_split.get()
            seed = self.var_seed.get()
            negs = self.var_negs.get().strip() or None

            yaml_path = build_yolo_seg_from_labelstudio(
                json_path,
                imgs_path,
                str(out_dataset),
                classes,
                allowed_labels=classes,
                allowed_geoms=geoms,
                val_split=split,
                seed=seed,
                negatives_dir=negs
            )

            # Validar
            if not (Path(yaml_path).parent / "images" / "train").exists():
                 messagebox.showerror("Error", "No se generó images/train.")
                 return
                 
            self.var_yaml.set(f"OK: {yaml_path}")
            self._log(f"Dataset preparado en {out_dataset}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _auto_adjust_params(self, event=None):
        sz = self.var_size.get()
        if "l (" in sz or "x (" in sz:
             # Check if current batch is just a number > 2
             try:
                 val = float(self.var_batch.get())
                 if val > 2:
                     self.var_batch.set("2")
                     self._log("SMART: Batch reducido a 2 (Modelo L/X).")
             except: pass

    def _is_detection_running(self) -> bool:
        try:
            import psutil
            for p in psutil.process_iter(['name', 'cmdline']):
                if p.info['name'] and 'python' in p.info['name'].lower():
                    if p.info['cmdline'] and any("detect_manchas" in a for a in p.info['cmdline']):
                        return True
        except ImportError:
            try:
                cmd = ["wmic", "process", "where", "name='python.exe'", "get", "commandline"]
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="cp850")
                if "detect_manchas" in proc.stdout: return True
            except: pass
        return False

    def _train_async(self):
        if self._is_detection_running():
            messagebox.showerror("CRÍTICO", "Se detectó 'detect_manchas' corriendo. Cierra la otra ventana.")
            return

        if not self.var_yaml.get().startswith("OK: "):
            messagebox.showwarning("Dataset", "Prepara el dataset primero.")
            return

        yaml = self.var_yaml.get()[4:]
        run_dir = self.var_run_dir.get().strip() or None
        
        t = threading.Thread(
            target=train_yolo_seg,
            args=(
                yaml,
                self.var_imgsz.get(),
                self.var_epochs.get(),
                self.var_device.get(),
                self.var_version.get(),
                self.var_size.get(),
                self.var_batch.get(),
                self.var_workers.get(),
                self.log_q,
                run_dir,
                self.var_mask_ratio.get(),
                self.var_overlap_mask.get(),
            ),
            daemon=True,
        )
        t.start()

    def _export_async(self):
        if not self.var_ckpt.get(): return
        t = threading.Thread(
            target=export_trt,
            args=(self.var_ckpt.get(), self.var_imgsz.get(), self.var_device.get(), self.log_q),
            daemon=True,
        )
        t.start()

    def _drain_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg.startswith("TRAIN_DONE:"):
                    self.var_last_run.set(f"runs dir: {msg.split(':', 1)[1]}")
                self._log(msg)
        except queue.Empty: pass
        self.after(100, self._drain_logs)

    def _log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")

    def _apply_recommended_v26(self):
        self.var_version.set("v26")
        self.var_size.set("s (small)")
        self.var_imgsz.set(1280)
        self.var_epochs.set(120)
        self.var_batch.set("8")
        self.var_workers.set(0)
        self._log("Preset applied: YOLO26 s | 1280 | 120ep | b=8")

    def _apply_low_vram(self):
        self.var_batch.set(2)
        self.var_workers.set(0)
        self._log("Preset Low VRAM: Batch=2, Workers=0. Safe for 1280px.")

    def _install_deps_async(self):
        def worker():
             self.log_q.put("Installing deps...")
             # ... simplified dep install command logic ...
             subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip", "ultralytics", "label-studio-converter", "opencv-python", "psutil"], capture_output=True)
             self.log_q.put("Deps OK (psutil included).")
        threading.Thread(target=worker, daemon=True).start()

    def _download_model_async(self):
         # Same logic as before but wrapped for threading
         def worker():
             self.log_q.put("Downloading model...")
             # Reuse _resolve_model_name and YOLO() download
             try:
                 ver, sz, model_name = _resolve_model_name(self.var_version.get(), self.var_size.get())
                 from ultralytics import YOLO
                 YOLO(model_name)
                 self.log_q.put(f"Downloaded {model_name}")
             except Exception as e:
                 self.log_q.put(f"Error: {e}")
         threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    App().mainloop()
