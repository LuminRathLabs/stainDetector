import os
import shutil
import datetime
import traceback

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(os.path.floor(os.path.log(size_bytes, 1024)))
    p = os.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def run_backup():
    # Rutas configuración
    source_dir = r"data"
    target_base_dir = r"\\fabrica\ia\Backup"
    target_actual_dir = os.path.join(target_base_dir, "Actual")
    log_file_path = os.path.join(target_base_dir, "backup_info.txt")
    
    status_message = ""
    success = False
    total_size_str = "0 B"
    
    try:
        # 1. Verificar que la carpeta destino base existe
        if not os.path.exists(target_base_dir):
            os.makedirs(target_base_dir, exist_ok=True)
            
        # 2. Si ya existe la carpeta "Actual" en el backup, la borramos
        if os.path.exists(target_actual_dir):
            shutil.rmtree(target_actual_dir)
            
        # 3. Copiar la carpeta
        shutil.copytree(source_dir, target_actual_dir)
        
        # 4. Calcular tamaño
        total_size = get_folder_size(target_actual_dir)
        # Manually implementing format_size since math.log might be missing if I don't import math
        import math
        if total_size == 0:
            total_size_str = "0 B"
        else:
            size_name = ("B", "KB", "MB", "GB", "TB")
            i = int(math.floor(math.log(total_size, 1024)))
            p = math.pow(1024, i)
            s = round(total_size / p, 2)
            total_size_str = f"{s} {size_name[i]}"
            
        success = True
        status_message = "Copia de seguridad realizada con éxito."
        
    except Exception as e:
        success = False
        status_message = f"Error durante la copia de seguridad: {str(e)}"
        print(status_message)
        traceback.print_exc()

    # 5. Escribir el log (txt)
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"Fecha y hora: {now}\n")
            f.write(f"Tamaño total: {total_size_str}\n")
            f.write(f"Estado: {'OK' if success else 'ERROR'}\n")
            f.write(f"Detalles: {status_message}\n")
    except Exception as log_error:
        print(f"No se pudo escribir el archivo de log: {str(log_error)}")

if __name__ == "__main__":
    run_backup()
