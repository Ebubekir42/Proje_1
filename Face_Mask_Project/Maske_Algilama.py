import numpy as np
import cv2
import imutils
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

class Maske_Algilama:

    def __init__(self,url = "0"):
        self.url = url
        self.yuz_agi = self.yuz_aginin_agirliklarini_yukleme()

        # Keras'tan eğitilmiş modeli yükleme
        self.maske_agi = load_model("mask_detector.model")
        self.ekran_görüntüsünü_gösterme()

    def ekran_görüntüsünü_gösterme(self):
        if ".png" in self.url:
            self.get_the_Image(self.url)
        elif "0" in self.url:
            self.url = int(self.url)
            self.get_the_Video(self.url)
        elif ".mp4" in self.url:
            self.get_the_Video(self.url)


    def get_the_Video(self,url):
        video = cv2.VideoCapture(url)
        while 1:

            # Görüntü akışından bir kare okuma
            durum,cerceve = video.read()
            # Görüntüyü yeniden boyutlandırma
            cerceve = imutils.resize(cerceve,width=400,height=400)
            konumlar,maske_algilama_tahmini = self.maske_algılama_ve_tahmin_etme(cerceve,self.yuz_agi,self.maske_agi)

            for (yuz_cercevesi, maske_algilama_tahmini) in zip(konumlar, maske_algilama_tahmini):
                self.ekran_goruntusunun_uzerine_yazma(cerceve,yuz_cercevesi,maske_algilama_tahmini)
            cv2.imshow("Maske Algilama", cerceve)

            # Çıkış yapmak için "Esc" tuşuna bas
            if  cv2.waitKey(10) == 27:
                break
        video.release()
        cv2.destroyAllWindows()

    def get_the_Image(self,url):

        while 1:

            # Görüntü akışından bir kare okuma
            cerceve = cv2.imread(url)
            # Görüntüyü yeniden boyutlandırma
            cerceve = imutils.resize(cerceve,width=400,height=400)
            konumlar,maske_algilama_tahmini = self.maske_algılama_ve_tahmin_etme(cerceve,self.yuz_agi,self.maske_agi)

            for (yuz_cercevesi, maske_algilama_tahmini) in zip(konumlar, maske_algilama_tahmini):
                self.ekran_goruntusunun_uzerine_yazma(cerceve, yuz_cercevesi, maske_algilama_tahmini)
            cv2.imshow("Maske Algilama", cerceve)

            # Çıkış yapmak için "Esc" tuşuna bas
            if  cv2.waitKey(10) == 27:
                break

    def ekran_goruntusunun_uzerine_yazma(self,cerceve,yuz_cercevesi,maske_algilama_tahmini):

        baslangic_X, baslangic_Y, bitis_X, bitis_Y = yuz_cercevesi

        # "Maske" + "Maske Olmayan" = 1
        maske,no_maske = maske_algilama_tahmini

        # Maske var/yok etiketini yazdırma
        etiket = "Maske Var" if maske > no_maske else "Maske Yok"

        # BGR -> Blue,Green,Red  (0,255,0) -> Yeşil  (0,0,255) -> Kırmızı
        renk = (0,255,0) if etiket == "Maske Var" else (0,0,255)

        # Video üzerine metin yazma
        cv2.putText(cerceve,etiket,(baslangic_X,baslangic_Y-4),cv2.FONT_HERSHEY_PLAIN,1.3,renk,2)

        # Video üzerine dikdörtgen çizme
        cv2.rectangle(cerceve,(baslangic_X,baslangic_Y),(bitis_X,bitis_Y),renk,2)

    def yuz_aginin_agirliklarini_yukleme(self):
        yapilandirma = "deploy.protext"
        model = "res10_300x300_ssd_iter_140000.caffemodel"
        return cv2.dnn.readNet(model, yapilandirma)   #dnn = deep neural networks = derin sinir ağı

    # Bu fonksiyonun amacı: yüz çerçevesinin konumları ve maske algılama tahmini döndürmek için
    def maske_algılama_ve_tahmin_etme(self,cerceve,yuz_agi,maske_agi):

        (yukseklik,genislik) = cerceve.shape[:2]  # cerceve.shape[0] => çerçevenin yüksekliği
                                                # cereceve.shape[1] => çerçevenin genişliği
        # Görüntüden veri bloğu oluşturma
        veri_blogu = cv2.dnn.blobFromImage(cerceve, 1.0, (224,224), (104.0,177.0,123.0))

        # Yüz ağının veri bloğunu ayarlaması
        yuz_agi.setInput(veri_blogu)

        # Yüz ağının ileri beslemeli işlem yapması
        algilama = yuz_agi.forward()
        # 4 boyutlu bir matris elde edilir.

        yuzler = []
        konumlar = []
        maske_algilama_tahmini = []

        for i in range(0,algilama.shape[2]):   # algilama.shape[2] => yüz ağının sayısı kadar
            yuz_algilama_tahimini = algilama[0,0,i,2]

            # Yüz algılama tahiminin değeri 0.5 ten büyük ise yüz algılıyor demektir.
            if yuz_algilama_tahimini > 0.5:

                box = algilama[0, 0, i, 3:7] * np.array([genislik, yukseklik, genislik, yukseklik])
                # algilama[0, 0, i, 3] = baslangic_X    BUnların değeri 0 ile 1 arasıdır. Yüz çerçevenin
                # algilama[0, 0, i, 4] = baslangic_Y       konumları belirlemek için x * genişlik
                # algilama[0, 0, i, 5] = bitis_X                                     y * yüksekilk
                # algilama[0, 0, i, 6] = bitis_Y

                # Başlangıçta tüm değişkenlerin veri tipi floattır. int e çevirmek için
                (baslangic_X, baslangic_Y, bitis_X, bitis_Y) = box.astype("int")
                (baslangic_X,baslangic_Y) = (max(0,baslangic_X), max(0, baslangic_Y))
                (bitis_X, bitis_Y) = (min(genislik - 1, bitis_X), min(yukseklik - 1, bitis_Y))


                yuz = cerceve[baslangic_Y:bitis_Y,baslangic_X:bitis_X]

                # Görüntünün boyutlarını değiştirme
                yuz = cv2.resize(yuz,(224,224))

                # Görüntünün Numpy dizisine dönüştürme
                yuz = np.array(yuz)

                # Görüntüyü ön işlemek için
                yuz = preprocess_input(yuz)
                yuzler.append(yuz)
                konumlar.append((baslangic_X,baslangic_Y,bitis_X,bitis_Y))

        if len(yuzler) > 0:
            yuzler = np.array(yuzler)

            # Tahmin yapmak için  (mask,no_mask)
            maske_algilama_tahmini = maske_agi.predict(yuzler)

        return (konumlar,maske_algilama_tahmini)

# p = Maske_Algilama("maskesiz.png")
# p = Maske_Algilama("maskeli.png")
# p = Maske_Algilama("....mp4")
p = Maske_Algilama() # Kamera açmak için

