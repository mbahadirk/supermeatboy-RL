# 2-10 toplama denetimi — 12 Kasım 2026

Yeni 70 dosya, önceki `collect_2-10_ppo50_01` kaynağına wave_00373…wave_00442 olarak eklenmiş. Tamamı çevrimdışı okundu; son dosya kısmi. Kaynak kayıtlara yazılmadı.

| Ölçü | Sonuç |
| --- | ---: |
| Durum satırı | 1.449.419 |
| Ardışık geçiş | 1.445.919 |
| Tamamlanan dalgalarda ajan izi / birebir farklı iz | 3.450 / 3.450 |
| Anahtar maskesi 0 → 1 | 2.034 |
| Bunların anahtara 30 birim içinde gerçekleşmesi | 2.034 |
| Anahtarlı durumdan başlayan geçiş | 620.861 |
| Anahtarı alıp daha sonra kilit bölgesine giren ajan kesiti | 517 |
| Son satırı bitiş olan kesit | 512 |
| Son satırı ölüm olan kesit | 2.945 |
| Son satırı yaşayan kesit | 43 |
| Fan etkili geçiş | 473.777 |
| Fan bağlamı bilinen geçiş | 1.445.919 |

Anahtar (69,7605;130,0100), kayıttaki uygulanmış lock_context grubunun hücresi 819. Kilit bölgesi x[-20,0], y[100,120] olarak alındı. Geçiş sayımı, ardışık kaydedilmiş x/y noktalarını bağlayan segmentin bu dikdörtgenle kesişmesidir. Bu, karakterin tam çarpışma gövdesinin kontrollü testi değildir.

**Önemli sınır:** Anahtarsız 15 segment de bu bölgeyle kesişiyor; bunlar üç ajan kesitinde, çoğunlukla üst kenara yakın kaydedilmiş noktalarda bulunuyor. Dolayısıyla 'anahtarı olmayan ajanlar kesinlikle durduruluyor' iddiası bu denetimle doğrulanmadı. Yerel JS kodu kilitleri her ajanın maskesine göre güncellemeden önce açıp sonra geri koyuyor; kayıtlı nokta ile çarpışma gövdesi ilişkisi ayrıca incelenmeden bu 15 segmenti hatalı veri olarak silmedik.

Geçişler, önceki fan/anahtar veri açığını gerçekten azaltıyor. Birebir farklı tam iz sayısı, ortak başlangıçların veya yakın benzer izlerin bağımsız olduğu anlamına gelmez. Fan maruziyeti ortak `wm.features.wind` hesabıdır; kuvvetin bağımsız fizik ölçümü değildir.

Ham sonuç ve dosya hash'leri `key_coverage.json`, bütün güncel kaynak envanteri `inventory.json`, tekrar üretim `audit.py` içindedir. Yeni toplam kaynak 443 dosya ve 8.631.891 satırdır; önceki 373 dosyalık rapor tarihsel bir snapshot olarak korunmuştur.
