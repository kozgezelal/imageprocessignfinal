# UYG332 Image Processing Final Project
** [ÖZGE ZELAL KÜÇÜK]  
** [B2180.060052]  

## Gereksinimler
- Python 3.8+
- Gerekli kütüphaneler:
  ```bash
  #%% [markdown]
# # UYG332 Image Processing Final Project
# **Öğrenci Adı:** [ÖZGE ZELAL KÜÇÜK]
# **Öğrenci No:** [B2180.060052]
#
# ## Problem 1: https://github.com/kozgezelal/imageprocessignimages/blob/main/tf2_engineer.jpg

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# 1. Orijinal resmi okuyıp ve gösteriyorum
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/tf2_engineer.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,6))
plt.imshow(img_rgb)
plt.title('https://github.com/kozgezelal/imageprocessignimages/blob/main/tf2_engineer.jpg (Problem 1)')
plt.axis('off')
plt.show()

# 2. Merkez noktayı bul ve yoğunluk değerini yazdırıyorum
height, width, _ = img.shape
yc, xc = height//2, width//2
print(f"Merkez Koordinatları: (y={yc}, x={xc})")
print(f"Merkez Yoğunluk Değeri (BGR): {img[yc, xc]}")

# 3. Renkli dikdörtgen ekle (30x40 piksel, #329ea8)
color_patch = np.zeros((30, 40, 3), dtype=np.uint8)
color_patch[:] = [168, 158, 50]  # Hex #329ea8 → BGR(168,158,50)

img_patched = img.copy()
y_start = yc - 15
y_end = yc + 15
x_start = xc - 20
x_end = xc + 20
img_patched[y_start:y_end, x_start:x_end] = color_patch

# 4. Yeni merkez yoğunluğunu yazdırıyorum
print(f"Yama Sonrası Merkez Değeri: {img_patched[yc, xc]}")

# 5. Düzenlenmiş resmi gösteriyorum
img_patched_rgb = cv2.cvtColor(img_patched, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,6))
plt.imshow(img_patched_rgb)
plt.title('Renk Yamalı Resim (Problem 1)')
plt.axis('off')
plt.show()

#%% [markdown]
# ## Problem 2: https://github.com/kozgezelal/imageprocessignimages/blob/main/einstein%20(1).tif

#%%
# 1. Gri tonlamalı olarak okudu ve gösterdi
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/einstein%20(1).tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.title('Orijinal Gri Einstein (Problem 2)')
plt.axis('off')
plt.show()

# 2-3. Negatifini alıp ve gösterdi
negative = 255 - img
plt.figure(figsize=(8,6))
plt.imshow(negative, cmap='gray')
plt.title('Negatif Resim (Problem 2)')
plt.axis('off')
plt.show()

# 4. Rastgele 5 pikseli karşılaştır
import random
random.seed(42)  # Tekrarlanabilirlik için

pixels = []
for _ in range(5):
    y = random.randint(0, img.shape[0]-1)
    x = random.randint(0, img.shape[1]-1)
    pixels.append((y, x))

print("(y,x)\tOrijinal\tNegatif")
for (y, x) in pixels:
    print(f"({y},{x})\t{img[y,x]}\t\t{negative[y,x]}")

#%% [markdown]
# ## Problem 3: https://github.com/kozgezelal/imageprocessignimages/blob/main/pout.tif

#%%
# 1. Gri tonlamalı oku ve göster
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/pout.tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.title('Orijinal Pout Resmi (Problem 3)')
plt.axis('off')
plt.show()

# 2. Log dönüşümü uyguladım
c = 255 / np.log(1 + 255)
log_transformed = c * np.log1p(img.astype(np.float32))
log_transformed = np.uint8(log_transformed)

plt.figure(figsize=(8,6))
plt.imshow(log_transformed, cmap='gray')
plt.title('Log Dönüşümü (Problem 3)')
plt.axis('off')
plt.show()

# 3. Ters log dönüşümü (orijinale uygula)
inv_log = np.exp(img.astype(np.float32) - 1
inv_log = np.uint8(cv2.normalize(inv_log, None, 0, 255, cv2.NORM_MINMAX))

plt.figure(figsize=(8,6))
plt.imshow(inv_log, cmap='gray')
plt.title('Ters Log Dönüşümü (Orijinale Uygulandı) (Problem 3)')
plt.axis('off')
plt.show()

# 4. Log uygulanmış resme ters log uyguluyorum
restored = np.exp(log_transformed.astype(np.float32)) - 1
restored = np.uint8(cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX))

plt.figure(figsize=(8,6))
plt.imshow(restored, cmap='gray')
plt.title('Log + Ters Log ile Geri Yükleme (Problem 3)')
plt.axis('off')
plt.show()

# 5. Yorumlar
print("""
YORUMLAR:
1. Log Dönüşümü: Karanlık bölgelerdeki detayları belirginleştiriyor
2. Ters Log (Orijinale): Parlak bölgeleri aşırı vurgular, doğal görünüm kaybı
3. Geri Yükleme: Log + Ters Log işlemi orijinale çok yakın sonuç verir
   (küçük sayısal hatalar dışında)
""")

#%% [markdown]
# ## Problem 4: https://github.com/kozgezelal/imageprocessignimages/blob/main/moon.tif

#%%
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# 1. Gri tonlamalı okuyup ve gösteriyorum
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/moon.tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.title('Orijinal Ay Resmi (Problem 4)')
plt.axis('off')
plt.show()

# Uzamsal Bölge Keskinleştirme Fonksiyonu
def spatial_unsharp(img, k):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    mask = img.astype(np.float32) - blur.astype(np.float32)
    return np.clip(img + k * mask, 0, 255).astype(np.uint8)

# Frekans Bölgesi Keskinleştirme Fonksiyonu
def frequency_unsharp(img, k, D0=30):
    f = fft2(img.astype(np.float32))
    fshift = fftshift(f)

    rows, cols = img.shape
    y, x = np.ogrid[:rows, :cols]
    center = (rows//2, cols//2)
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Gaussian yüksek geçiren filtre
    H = 1 - np.exp(-(dist**2)/(2*(D0**2)))

    filtered = fshift * (1 + k * H)
    f_ishift = ifftshift(filtered)
    result = np.abs(ifft2(f_ishift))
    return np.uint8(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX))

# 2-3. Farklı k değerleri için işlem
k_values = [0.5, 1.0, 1.5]
plt.figure(figsize=(15,10))

for i, k in enumerate(k_values):
    # Uzamsal bölge
    spatial = spatial_unsharp(img, k)

    # Frekans bölgesi
    freq = frequency_unsharp(img, k)

    # Görselleştirme
    plt.subplot(3, 2, 2*i+1)
    plt.imshow(spatial, cmap='gray')
    plt.title(f'Uzamsal (k={k})')
    plt.axis('off')

    plt.subplot(3, 2, 2*i+2)
    plt.imshow(freq, cmap='gray')
    plt.title(f'Frekans (k={k}, D0=30)')
    plt.axis('off')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## Problem 5: https://github.com/kozgezelal/imageprocessignimages/blob/main/pcb.tif

#%%
# 1. Gri tonlamalı okuyorum ve gösteriyorum
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/pcb.tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.title('Orijinal PCB Resmi (Problem 5)')
plt.axis('off')
plt.show()

# 2. Gürültü analizi
plt.figure(figsize=(8,4))
plt.hist(img.ravel(), 256, [0,256])
plt.title('Histogram - Gürültü Analizi (Problem 5)')
plt.show()

print("""
GÜRÜLTÜ ANALİZİ:
- Tuz-biber gürültüsü (siyah ve beyaz noktalar)
- Histogramda 0 ve 255'te yoğunlaşma
- Rastgele dağılmış yüksek kontrastlı pikseller
""")

# 3. Gürültü temizleme
denoised = cv2.medianBlur(img, 3)  # 3x3 medyan filtresi

plt.figure(figsize=(12,5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Gürültülü Resim')
plt.subplot(122), plt.imshow(denoised, cmap='gray'), plt.title('Temizlenmiş Resim (Medyan Filtre)')
plt.show()

#%% [markdown]
# ## Problem 6: https://github.com/kozgezelal/imageprocessignimages/blob/main/pollen.tif

#%%
# 1. Gri tonlamalı okuyup ve gösteriyorum
img = cv2.imread(IMAGE_PATH + 'https://github.com/kozgezelal/imageprocessignimages/blob/main/pollen.tif', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8,6))
plt.imshow(img, cmap='gray')
plt.title('Orijinal Polen Resmi (Problem 6)')
plt.axis('off')
plt.show()

# 2. Problem analizi
plt.figure(figsize=(10,4))
plt.subplot(121), plt.hist(img.ravel(), 256, [0,256]), plt.title('Histogram (Problem 6)')
plt.subplot(122), plt.imshow(img, cmap='gray'), plt.title('Orijinal')
plt.show()

print("""
PROBLEM ANALİZİ:
- Düşük kontrast (histogram dar aralıkta)
- Detaylar belirsiz (pikseller 80-150 arasında yoğunlaşmış)
- Standart sapma: {:.1f} (düşük kontrastı gösterir)
""".format(img.std()))

# 3. Çözüm 1: Histogram eşitlemem
equalized = cv2.equalizeHist(img)

# Çözüm 2: Kontrast germem
min_val = np.min(img)
max_val = np.max(img)
stretched = cv2.convertScaleAbs(img, alpha=255/(max_val-min_val),
                               beta=-min_val*255/(max_val-min_val))

# 4. Sonuçları gösteriyorum
plt.figure(figsize=(15,5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Orijinal')
plt.subplot(132), plt.imshow(equalized, cmap='gray'), plt.title('Histogram Eşitleme')
plt.subplot(133), plt.imshow(stretched, cmap='gray'), plt.title('Kontrast Germe')
plt.show()

# 5. Nicel karşılaştırmam
plt.figure(figsize=(15,4))
plt.subplot(131), plt.hist(img.ravel(), 256, [0,256]), plt.title('Orijinal Histogram')
plt.subplot(132), plt.hist(equalized.ravel(), 256, [0,256]), plt.title('Eşitlenmiş Histogram')
plt.subplot(133), plt.hist(stretched.ravel(), 256, [0,256]), plt.title('Gerilmiş Histogram')
plt.show()

print("""
ÇÖZÜM GEREKÇELENDİRME:
- Histogram eşitleme: Standart sapma {:.1f} → {:.1f} ({:.1f}x artış)
- Kontrast germe: Standart sapma {:.1f} → {:.1f} ({:.1f}x artış)
- Her iki yöntem de kontrastı belirgin şekilde iyileştirdi
- Histogram eşitleme daha agresif, kontrast germe daha doğal sonuç verdi
""".format(
    img.std(), equalized.std(), equalized.std()/img.std(),
    img.std(), stretched.std(), stretched.std()/img.std()
))
  pip install opencv-python numpy matplotlib scipy
