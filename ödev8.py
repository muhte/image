import cv2
import numpy as np

def restore_image(image_path):
    # Görüntüyü oku
    image = cv2.imread(image_path)
    
    # Görüntüyü gri tonlamaya dönüştür
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Maske oluştur (örneğin, gri tonlamada 0 olmayan her pikseli içeren bir maske)
    mask = (gray_image > 0).astype(np.uint8) * 255

    # Inpainting işlemi
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # Görüntüyü görselleştir
    cv2.imshow('Restored Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Görüntü yolunu belirt
image_path = '/Users/muhteber/Desktop/image/static/oldphoto/photo3.jpeg'

# Görüntüyü restore et
restore_image(image_path)
