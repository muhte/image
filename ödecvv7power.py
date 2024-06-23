

#^//*power log transform işlemi yaopılıyorr*//


import cv2
import matplotlib.pyplot as plt
import numpy as np

def erode_clahe_power_log_transform(img_path, gamma=0.5):
    # Resmi oku
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Erozyon işlemi uygula
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=1)

    # CLAHE uygulama
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(cv2.cvtColor(eroded_img, cv2.COLOR_RGB2GRAY))

    # Power Log Transform uygula
    power_log_transformed = np.power(clahe_img / 255.0, gamma)

    # Orijinal, erozyon uygulanmış, CLAHE uygulanmış ve power log transform'u görselleştirme
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Orijinal')

    plt.subplot(1, 4, 2)
    plt.imshow(eroded_img)
    plt.title('Erozyon')

    plt.subplot(1, 4, 3)
    plt.imshow(clahe_img, cmap='gray')
    plt.title('CLAHE')

    plt.subplot(1, 4, 4)
    plt.imshow(power_log_transformed, cmap='gray')
    plt.title('Power Log Transform')

    plt.show()

# Fonksiyonu çağırma
erode_clahe_power_log_transform("/imagepath", gamma=0.5)
