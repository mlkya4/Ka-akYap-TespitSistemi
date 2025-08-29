Kaçak Yapı / Afet Etki Tespit Sistemi (Streamlit)

Bu proje; uydu görüntülerini periyodik olarak alıp veritabanına kaydeder, daha sonra farklı tarihlerdeki görüntüleri karşılaştırarak:

🧱 Kaçak yapı tespiti

🔥 Yangın/yanıklık alan tespiti (RGB’de basit göstergeler; kurumsal kullanımda uygun spektral/termal kaynaklarla güçlendirilmeli)

🌪️ Deprem/sel vb. afet sonrası hasar tespiti

gibi senaryolarda değişim alanlarını ortaya çıkarır.

Özellikler

📥 Görüntü edinme: Google Static Maps (veya diğer kaynaklar) üzerinden koordinat/zoom ile çekim

🗄️ Veritabanı kaydı: Görüntü, meta veriler ve kullanıcılar PostgreSQL’de saklanır

🧠 Segmentasyon & değişim analizi: torchvision FCN-ResNet50 ile temel yüzey/alan ayrımı + iki tarih arasındaki fark haritası

🖼️ Görselleştirme: Isı haritası, kontur/kare çizimleri, metrikler (değişim yüzdesi, alan, yapı sayısı vb.)

🖥️ Streamlit arayüzü: Koordinat seçimi, analiz başlatma ve raporlama sekmeleri
<img width="800" height="800" alt="old_segmentation" src="https://github.com/user-attachments/assets/e5beed36-cb1a-46d4-a28f-d96115542366" />
