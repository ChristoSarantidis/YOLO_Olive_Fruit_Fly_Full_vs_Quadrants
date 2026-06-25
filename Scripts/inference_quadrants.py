import os
import glob
import cv2
from ultralytics import YOLO
import csv
import psutil
import subprocess
from datetime import datetime
import time
import resend

# ---- ΡΥΘΜΙΣΕΙΣ ΧΡΗΣΤΗ ----
model_path = "best.pt"
input_folder = "images"  # Ο φάκελος με τις αρχικές εικόνες
output_root = "output_dataset"   # Ο γενικός φάκελος εξόδου
valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SAVE_ANNOTATED_IMAGES = True     # True: Σώζει την εικόνα με τα κουτάκια
SAVE_EMPTY_LABELS = True        # False: Αν δεν βρει τίποτα, δεν φτιάχνει txt αρχείο
IMGSZ = 1664
CONF_THRESHOLD = 0.35            # Όριο εμπιστοσύνης (προαιρετικό)
# --------------------------

# 1. Δημιουργία δομής φακέλων (images / labels)
img_out_dir = os.path.join(output_root, "images")
lbl_out_dir = os.path.join(output_root, "labels")
perf_log_dir = os.path.join(output_root, "performance_logs") # Φάκελος για logs επιδόσεων

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(lbl_out_dir, exist_ok=True)
os.makedirs(perf_log_dir, exist_ok=True) # Δημιουργία φακέλου για logs

print(f"📂 Output Folders:\n  - {img_out_dir}\n  - {lbl_out_dir}\n  - {perf_log_dir}")


# Ορισμός αρχείων CSV για καταγραφή επιδόσεων
cpu_log_file = os.path.join(perf_log_dir, "cpu_performance.csv")
temp_log_file = os.path.join(perf_log_dir, "temperature.csv")
memory_log_file = os.path.join(perf_log_dir, "memory_usage.csv")
disk_log_file = os.path.join(perf_log_dir, "disk_io.csv")
inference_log_file = os.path.join(perf_log_dir, "inference_times.csv")


# Helper function για εγγραφή σε CSV
def write_to_csv(filepath, data, fieldnames):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)



# Φόρτωση μοντέλου
model = YOLO(model_path)

def split_into_quadrants(img):
    """Κόβει την εικόνα σε 4 ίσα μέρη (TL, TR, BL, BR)."""
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    return [
        (img[0:cy, 0:cx], "tile1_TL"), # Top-Left
        (img[0:cy, cx:W], "tile2_TR"), # Top-Right
        (img[cy:H, 0:cx], "tile3_BL"), # Bottom-Left
        (img[cy:H, cx:W], "tile4_BR"), # Bottom-Right
    ]

# --- Functions for Performance Monitoring ---
def get_cpu_metrics():
    """Επιστρέφει μετρήσεις CPU: χρήση και load average."""
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
    load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0) # Linux-specific
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu_percent,
        "load_avg_1min": load_avg[0],
        "load_avg_5min": load_avg[1],
        "load_avg_15min": load_avg[2],
    }

def get_memory_metrics():
    """Επιστρέφει μετρήσεις χρήσης μνήμης."""
    mem = psutil.virtual_memory()
    return {
        "timestamp": datetime.now().isoformat(),
        "total_mb": round(mem.total / (1024 * 1024), 2),
        "available_mb": round(mem.available / (1024 * 1024), 2),
        "percent": mem.percent,
        "used_mb": round(mem.used / (1024 * 1024), 2),
        "free_mb": round(mem.free / (1024 * 1024), 2),
    }

def get_disk_io_metrics():
    """Επιστρέφει μετρήσεις Disk I/O."""
    disk_io = psutil.disk_io_counters()
    return {
        "timestamp": datetime.now().isoformat(),
        "read_count": disk_io.read_count,
        "write_count": disk_io.write_count,
        "read_bytes_mb": round(disk_io.read_bytes / (1024 * 1024), 2),
        "write_bytes_mb": round(disk_io.write_bytes / (1024 * 1024), 2),
    }

def get_cpu_temperature():
    """Επιστρέφει τη θερμοκρασία CPU για Raspberry Pi."""
    try:
        # vcgencmd is specific to Raspberry Pi
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp_str = output.split("=")[1].split("'")[0]
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature_celsius": float(temp_str)
        }
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        # Fallback for non-Pi systems or if vcgencmd fails
        try:
            # Try reading from sysfs, common on Linux
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_raw = f.read().strip()
                temperature_celsius = float(temp_raw) / 1000.0
                return {
                    "timestamp": datetime.now().isoformat(),
                    "temperature_celsius": temperature_celsius
                }
        except (FileNotFoundError, ValueError):
            return {
                "timestamp": datetime.now().isoformat(),
                "temperature_celsius": "N/A"
            }

# Σημείωση για την κατανάλωση ρεύματος:

# Η άμεση λογισμική μέτρηση της κατανάλωσης ρεύματος σε Raspberry Pi χωρίς

# επιπρόσθετο υλικό (π.χ. αισθητήρες INA219) δεν είναι εφικτή ή ακριβής.

# Για αξιόπιστες μετρήσεις απαιτείται εξωτερικό hardware.

# Ωστόσο, μπορούμε να κάνουμε μια εκτίμηση με βάση τη χρήση της CPU.



# Τιμές για εκτίμηση κατανάλωσης ρεύματος (για Raspberry Pi 4 Model B)

# Αυτές οι τιμές είναι ενδεικτικές και μπορεί να διαφέρουν ανάλογα με την έκδοση

# του Pi, τα συνδεδεμένα περιφερειακά και τον ακριβή φόρτο εργασίας.

POWER_IDLE_WATTS = 2.7  # Watts at 0% CPU usage

POWER_MAX_LOAD_WATTS = 6.4 # Watts at 100% CPU usage (full stress)



def estimate_power_consumption(cpu_percent):

    """

    Εκτιμά την κατανάλωση ρεύματος (σε Watts) με βάση τη χρήση της CPU

    χρησιμοποιώντας ένα γραμμικό μοντέλο.

    """

    if not isinstance(cpu_percent, (int, float)):

        return "N/A"

    

    estimated_watts = POWER_IDLE_WATTS + (POWER_MAX_LOAD_WATTS - POWER_IDLE_WATTS) * (cpu_percent / 100.0)

    return round(estimated_watts, 2)



# Εύρεση εικόνων

image_paths = []

for ext in valid_exts:

    image_paths.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))

image_paths.sort()



if not image_paths:

    print("🚫 Δεν βρέθηκαν εικόνες στον φάκελο εισόδου.")

    exit()



print(f"🔎 Βρέθηκαν {len(image_paths)} εικόνες για επεξεργασία.")



# Κύριος βρόχος επεξεργασίας

for img_path in image_paths:

    stem = os.path.splitext(os.path.basename(img_path))[0]

    original_img = cv2.imread(img_path)



    if original_img is None:

        print(f"❌ Error reading: {img_path}")

        continue



    # Διαίρεση σε tiles
    tiles_info = split_into_quadrants(original_img)

    for tile_img, tile_suffix in tiles_info:
        # Μοναδικό όνομα αρχείου: originalName_tileX_Position
        unique_name = f"{stem}_{tile_suffix}"

        start_time = time.time() # Έναρξη μέτρησης χρόνου
        # Εκτέλεση πρόβλεψης (χωρίς αυτόματο save του YOLO για να έχουμε έλεγχο)
        results = model.predict(
            source=tile_img,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            device="cpu",   # Προσαρμογή για το Pi
            verbose=True,
            save_txt=True,
            save_conf=True
        )
        end_time = time.time() # Λήξη μέτρησης χρόνου
        inference_duration = end_time - start_time

        # Καταγραφή χρόνου inference
        inference_data = {
            "timestamp": datetime.now().isoformat(),
            "image_name": unique_name,
            "inference_time_sec": round(inference_duration, 4)
        }
        write_to_csv(inference_log_file, inference_data, list(inference_data.keys()))
        
        result = results[0] # Παίρνουμε το αποτέλεσμα της μίας εικόνας (tile)

        # --- Αποθήκευση Labels (.txt) ---
        # Ελέγχουμε αν βρήκε κάτι ή αν θέλουμε και κενά αρχεία
        if len(result.boxes) > 0 or SAVE_EMPTY_LABELS:
            txt_path = os.path.join(lbl_out_dir, f"{unique_name}.txt")
            
            with open(txt_path, "w") as f:
                for box in result.boxes:
                    # YOLO Format: class x_center y_center width height (normalized)
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist() 
                    conf = float(box.conf[0])
                    
                    # Γράφουμε στο αρχείο (μπορείς να προσθέσεις και το conf αν θες)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f} \n")

        # --- Αποθήκευση Εικόνας (.jpg) ---
        if SAVE_ANNOTATED_IMAGES:
            # Το plot() ζωγραφίζει τα boxes πάνω στην εικόνα
            plotted_img = result.plot() 
            save_img_path = os.path.join(img_out_dir, f"{unique_name}.jpg")
            cv2.imwrite(save_img_path, plotted_img)
        
    print(f"✅ Processed: {stem}")
    
    print("sleeping")
    time.sleep(5*60)
    print("awake")

    
    # --- Καταγραφή Επιδόσεων μετά την επεξεργασία κάθε εικόνας ---
    cpu_data = get_cpu_metrics()
    write_to_csv(cpu_log_file, cpu_data, list(cpu_data.keys()))
    
    # Εκτίμηση κατανάλωσης ρεύματος
    estimated_power = estimate_power_consumption(cpu_data["cpu_percent"])
    power_log_data = {"timestamp": cpu_data["timestamp"], "estimated_watts": estimated_power}
    power_log_file = os.path.join(perf_log_dir, "estimated_power.csv")
    write_to_csv(power_log_file, power_log_data, list(power_log_data.keys()))
    
    temp_data = get_cpu_temperature()
    write_to_csv(temp_log_file, temp_data, list(temp_data.keys()))

    memory_data = get_memory_metrics()
    write_to_csv(memory_log_file, memory_data, list(memory_data.keys()))

    disk_io_data = get_disk_io_metrics()
    write_to_csv(disk_log_file, disk_io_data, list(disk_io_data.keys()))
    
    print(f"📊 Performance logged for {stem}")
    

print("🎉 Η διαδικασία ολοκληρώθηκε!")
print(f"📁 Εικόνες: {img_out_dir}")
print(f"📝 Labels:  {lbl_out_dir}")
print(f"📊 Performance Logs: {perf_log_dir}")
print(f"⏱️ Inference Times: {inference_log_file}")