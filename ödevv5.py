import cv2
import matplotlib.pyplot as plt
import numpy as np

def remove_lines(img_path):
    # Resmi oku
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Histogram eşitleme ve kontrast arttırma
    equalized = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    contrast_stretched = ((equalized - np.min(equalized)) / (np.max(equalized) - np.min(equalized))) * 255

    # Gri tonlamaya dönüştürme
    gray = cv2.cvtColor(contrast_stretched.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Gauss filtreleme
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Median filtreleme
    median_filtered = cv2.medianBlur(blurred, 5)

    # Maskeyi oluştur
    mask = np.zeros_like(median_filtered, dtype=np.uint8)
    mask[median_filtered < 200] = 255

    # Maskeyi uygun formata dönüştürme
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Bilateral filtre uygulama
    bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)

    # Görüntüleri görselleştirme
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Orijinal')

    plt.subplot(1, 4, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Histogram Eşitleme')

    plt.subplot(1, 4, 3)
    plt.imshow(median_filtered)
    plt.title('Gauss ve Median Filtreleme')

    plt.subplot(1, 4, 4)
    plt.imshow(bilateral_filtered)
    plt.title('Bilateral Filtreleme')

    plt.show()

# Fonksiyonu çağırma
remove_lines("/Users/muhteber/Desktop/image/static/oldphoto/photo4.webp")
