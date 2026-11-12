# Yeni PPO verisi — 12 Kasım 2026

İncelenen kaynak: `runs/collect_2-10_ppo50_01/wm_data`. Klasör adı kapsamı anlatmıyor: kayıtlar 1-1…1-14, 3-1 ve 3-2 bölümlerine ait. Kullanıcı 2-10'da henüz toplamadığını doğruladı.

373 dosyanın tamamının metadata/header envanteri ve bütün 7.182.472 durum satırı incelendi. Satır denetimi 66,63 saniye sürdü; 120 saniye / 8 milyon satır sınırının içinde tamamlandı. Analiz tek dosyayı bir seferde açtı; özellik matrisi veya eğitim modeli oluşturulmadı. Kaynaklara yazılmadı. Her shard, world ve statik geometri baytlarının hash'i `coverage.json` içinde tutuluyor.

| Ölçü | Sonuç |
| --- | ---: |
| Bölüm | 16 |
| Kayıt dosyası | 373 |
| Kısmi dosya | 17 |
| Durum satırı | 7.182.472 |
| Ardışık durum/eylem geçişi | 7.164.622 |
| Ajan kesiti | 17.850 |
| Tamamlanan kayıt dalgalarındaki ajan izi | 17.050 |
| Bunların farklı tam iz sayısı | 17.050 |
| Kısmi dalgalardaki ajan kesiti | 800 |
| Kısmi kesitlerde farklı iz | 707 |
| Son satırı bitiş olan kesit | 6.952 |
| Son satırı ölüm olan kesit | 8.884 |
| Son satırı hâlâ yaşayan kesit | 2.014 |
| Anahtar edinimi / anahtarlı başlangıç geçişi | 0 / 0 |
| Fan etkili geçiş / bilinen fan bağlamlı geçiş | 0 / 7.164.622 |
| Eğitim dışı ayrılmış bölüm | 0 |
| Geçersiz veya sabit 16,667 ms dışında dt | 0 |

Son durum sayıları kısmi kesitleri de içerir; yaşayanların tamamını ölüm veya timeout saymıyoruz. Dosyaların 341'inde 50, 32'sinde 25 ajan izi var. Meta veride bildirilen terminal durum ile son satır arasında gözlenen eksik terminal satırı veya çelişki yok. Bu kontrol, bütün oyun içi olayların bağımsız fizik doğrulaması değildir.

Tamamlanan izlerde birebir tekrar bulunmaması olumlu; yakın benzerlik, ortak başlangıç ve aynı dünya akışını paylaşma devam edebilir. Bu nedenle 17.050 bağımsız deney veya görülmemiş harita başarısı iddia etmiyoruz. Tekrar kimliği ortak `wm.dataset.trace_id` sahibiyle, geometri/durum/eylem üzerinden hesaplandı; agent/wave/reward farkı tek başına yeni iz üretmez.

## Bölüm başına sonuçlar

Bitiş/ölüm sütunları kısmi dalgalar dahil son satırlardan sayılmıştır. Diğer kesitler kayıt sonunda hâlâ yaşıyor.

| Bölüm | Dosya | Satır | Bitiş | Ölüm |
| --- | ---: | ---: | ---: | ---: |
| 1-1 | 38 | 383.741 | 1.848 | 0 |
| 1-2 | 44 | 950.396 | 1.953 | 0 |
| 1-3 | 20 | 124.178 | 748 | 203 |
| 1-4 | 16 | 598.101 | 591 | 0 |
| 1-5 | 9 | 314.039 | 300 | 0 |
| 1-6 | 10 | 298.156 | 131 | 280 |
| 1-7 | 25 | 293.694 | 960 | 240 |
| 1-8 | 5 | 28.637 | 167 | 33 |
| 1-9 | 8 | 147.270 | 21 | 341 |
| 1-10 | 5 | 73.435 | 85 | 127 |
| 1-11 | 57 | 676.336 | 22 | 2.807 |
| 1-12 | 20 | 400.583 | 0 | 949 |
| 1-13 | 22 | 335.075 | 116 | 931 |
| 1-14 | 11 | 581.947 | 0 | 232 |
| 3-1 | 69 | 1.933.565 | 10 | 2.399 |
| 3-2 | 14 | 43.319 | 0 | 342 |

## Anlamı ve sonraki çalışma

Bu toplama genel hareket/ölüm/bitiş verisini belirgin artırıyor. Fan ve anahtar/kilit açığı kapanmış değil; daha çok benzer veri toplamaktan önce 2-10'da bu geçişlere erişmek gerekiyor. Bitişi olmayan bölümlerde de eldeki hatalı hareketler değerlidir; ancak bunlar başarılı yaklaşımın yerini tutmaz.

Önerilen deney sırası: mevcut eski 321 girdili gösterim ve donmuş veri/roster ile 3/6/12 epoch karşılaştırması; sonra yeni verinin kontrollü bir manifest uzantısıyla etkisi. Eğik/yarım blokların 419 girdili yeni gösterimi ayrı bir etken olarak sınanabilir. Kullanıcı gösterim seçimini henüz yapmadan eğitim başlatılmadı. Ham kayıtlar yeni gösterime dönüştürülebilir; PPO toplamasında `--shape-inputs` olmaması world model için engel değildir.

Devralınan blok değişikliklerinin 17 odaklı testi geçti. İlk çalıştırmada 15 test geçti, iki test geçici dizinin üst klasörü bulunmadığından kurulamadı; üst klasör oluşturulup yalnız bu ikisi tekrar çalıştırıldı ve geçti. Testlerden biri küçük sentetik eğitim fixture'ı kullanır; gerçek toplama verisiyle yeni eğitim yapılmadı. Canlı oyun/monitör doğrulaması yapılmadı.

Ham sonuçlar: `inventory.json`, `coverage.json`; yeniden üretim: `audit.py`. Bu rapor yeni bir eğitim manifesti veya veri yeterliliği kararı değildir.
