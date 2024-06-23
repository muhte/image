import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = "//Users/muhteber/Desktop/image/static/oldphoto"
    
    # Görüntüyü oku
    imgpath = path + "/photo4.webp"
    img = cv2.imread(imgpath, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Görüntüyü gri tonlamaya dönüştür
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Histogram dengeleme işlemi uygula
    img_equ = cv2.equalizeHist(img_gray)

    # İnpainting işlemi için maske oluştur
    mask = np.zeros_like(img_equ)
    mask[100:300, 200:400] = 255  # Örnek bir maske oluştur (çiziklerin olduğu bölgeler)

    # İnpainting işlemi uygula
    img_inpainted = cv2.inpaint(img_equ, mask, 5, cv2.INPAINT_TELEA)

    # Görüntüleri görselleştirme
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 4, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(1, 4, 3)
    plt.imshow(img_equ, cmap='gray')
    plt.title('Histogram Equalized Image')

    plt.subplot(1, 4, 4)
    plt.imshow(img_inpainted, cmap='gray')
    plt.title('Inpainted Image')

    plt.show()

if __name__ == "__main__":
    main()
