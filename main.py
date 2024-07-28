import mss
import numpy as np
from PIL import Image
from yeelight import Bulb
import time
import threading
from sklearn.cluster import KMeans
from collections import Counter
from queue import Queue, Empty  # Add the missing imports

# Replace with the IP addresses of your Yeelight strips
strip_ips = ["192.168.34.5", "192.168.34.6", "192.168.34.22"]
strips = [Bulb(ip) for ip in strip_ips]

# Queue for handling strip updates
update_queue = Queue()


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
        return img


def get_dominant_color(image, k=4, image_processing_size=None):
    if image_processing_size is not None:
        image = image.resize(image_processing_size, Image.Resampling.LANCZOS)
    np_image = np.array(image)
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))

    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(np_image)
    label_counts = Counter(labels)
    dominant_color = kmeans.cluster_centers_[label_counts.most_common(1)[0][0]]

    return tuple(map(int, dominant_color))


def calculate_brightness(color):
    r, g, b = color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return int((brightness / 255) * 100)


def set_color_for_strip(strip, color, retry_count=3):
    for attempt in range(retry_count):
        try:
            strip.set_rgb(color[0], color[1], color[2])
            brightness = calculate_brightness(color)
            strip.set_brightness(brightness)
            return
        except Exception as e:
            print(f"Failed to set color for {strip} on attempt {attempt + 1}: {e}")
            if "quota exceeded" in str(e):
                time.sleep(1)
            else:
                break


def update_strip_colors():
    while True:
        try:
            strip, color = update_queue.get(timeout=1)
            set_color_for_strip(strip, color)
            update_queue.task_done()
        except Empty:
            continue


def set_strip_color(strip, color):
    color = tuple(map(int, color))
    update_queue.put((strip, color))


def main():
    threading.Thread(target=update_strip_colors, daemon=True).start()

    while True:
        try:
            screen_image = capture_screen()
            width, height = screen_image.size

            bottom_middle = screen_image.crop((width // 3, height * 2 // 3, width * 2 // 3, height))
            left_middle = screen_image.crop((0, height // 3, width // 3, height * 2 // 3))
            top_middle = screen_image.crop((width // 3, 0, width * 2 // 3, height // 3))

            bottom_color = get_dominant_color(bottom_middle, image_processing_size=(100, 100))
            left_color = get_dominant_color(left_middle, image_processing_size=(100, 100))
            top_color = get_dominant_color(top_middle, image_processing_size=(100, 100))

            set_strip_color(strips[0], bottom_color)
            set_strip_color(strips[1], left_color)
            set_strip_color(strips[2], top_color)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(2)  # Adjust the sleep time as needed


if __name__ == "__main__":
    main()
