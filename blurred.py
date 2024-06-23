import cv2
import numpy as np

# Görüntüyü oku (görüntü_yolu.jpg dosyanızın bulunduğu dizini doğru olarak belirtmelisiniz)
image = cv2.imread('//Users/muhteber/Desktop/image/static/oldphoto/photo3.jpeg')

# Erozyon için kernel oluştur
kernel = np.ones((5, 5), np.uint8)

# Erozyon işlemi
eroded_image = cv2.erode(image, kernel, iterations=1)

# Dilasyon işlemi
dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

# Gaussian blur uygula
blurred_image = cv2.GaussianBlur(dilated_image, (5, 5), 0)

# İşlemden geçirilmiş görüntüyü tek bir pencerede göster
combined_image = np.hstack((image, eroded_image, dilated_image, blurred_image))
cv2.imshow('Original vs Processed Images', combined_image)

# Pencereyi açık tutmak ve kullanıcının kapatmasını beklemek için bekleyin
cv2.waitKey(0)
cv2.destroyAllWindows()
