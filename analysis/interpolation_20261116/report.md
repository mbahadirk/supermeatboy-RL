# Ara kare tahminleri — 16 Kasım 2026

**Güncellik notu:** Aşağıdaki sonuçlar ilk değişikliğin tarihsel kaydıdır. `online_check06` sonrasında
ara-kare konum tetiklerini susturma davranışı geri alındı: başarısız zıplamayı da geciktirdiği görüldü.
Güncel düzeltme ve sınırları [yeni raporda](../interpolation_safety_20261116/report.md).
Buradaki tetik azalması bir başarı/FPS iyileşmesinin kanıtı değildir. Tarihsel denetim betiği farklı
kontrolcü koduyla bu raporu yeniden yazmayı reddeder.

İlk düzeltme, dört karelik world model'in doğrusal ara değerlerini doğrudan fizik tahmini gibi
değerlendiren konum sapması kontrolünü düzeltiyor. Model ağırlıkları ve eğitim değişmedi.

## Davranış

- `Plan.path_step`, yolun doğrudan tahmin edilen noktalarının aralığını taşır. Model tabanlı yolda
  modelin adımı, kaydedilmiş gerçek rotada ve bir-karelik modelde 1 kullanılır. Aralık planın başlangıcına
  göredir; global kare numarasına göre değildir.
- Dört-karelik tahminin konum sapması kontrolü +4, +8, +12… noktalarında yapılır. Aradaki doğrusal
  konumlar çizim ve mevcut yol/temas/tehlike kontrolleri için korunur. Duvar/zemin teması, tehlike,
  değişen döşemeler, hareket sonu, hedef/anahtar kaçırma ve periyodik karar kuralları değiştirilmedi.
- `predictions` ve yeni kayıtların konum hata raporu yalnızca doğrudan tahmin noktalarını içerir.
  Eski kayıtlarda bu alan yoksa analiz önceki davranışını korur; bu arşivler otomatik yeniden yorumlanmaz.
- Yalnızca konumdan anlaşılabilen gerçek sapma, sonraki doğrudan tahmine kadar en fazla üç kare
  gecikebilir (60 Hz oyun zamanında 50 ms). Bu bir model doğruluğu iyileştirmesi değildir.

## Sabit kayıt karşılaştırması

Kaynak: `runs/online_check05` içindeki **ilk 4-4 ziyaretinin 1–57. tamamlanmış denemeleri**.
Sonradan eklenen bölümler/denemeler dışarıda bırakıldı. Girdi dosyalarının ve seçili planların SHA-256
özetleri `report.json` içindedir. `audit.py` oyunu çalıştırmaz ve eğitim yapmaz.

Her plan için ilk dört karede yalnızca gerçekten planla aynı eylemler uygulanmışken karşılaştırma yapıldı.
Başlangıç konumu kayda uymayan planlar, eksik/ölü durumlar ve değişen eylem sonrası pencereler sayılmadı.
İki tarafta aynı hata eşiği ve kaydedilmiş platform konumları kullanıldı. Kontrol pencereleri örtüşebilir.

| Ölçüm | Önce | Sonra |
|---|---:|---:|
| İncelenen kare-kontrol sayısı | 19.533 | 19.533 |
| Konum sapması tetiği | 2.206 | 299 |
| En az bir tetik bulunan kontrol (tehlike hariç) | 3.394 | 1.917 |
| Temas tetiği bulunan kontrol | 1.242 | 1.242 |

Doğrudan tahmin noktasındaki **4.688 kontrolde bütün tetikler birebir aynı**.
5.896 planın 134'ü kaydedilmiş gerçek rotaydı; bunların ara-kare kontrolleri korunuyor.
Dört eylemi de takip edilmiş 4.152 model-tabanlı pencerede, uç nokta hatası 1 birimden küçükken
ara hata 3 birimi geçen **91 örnek** var. Örnekler JSON raporunda.

Bu sayılar **daha az gerçek karar, daha yüksek başarı veya FPS artışı ölçümü değildir**.
Yeni kararlar alınmış olsaydı izlenecek alternatif eylemler ve sonuçları burada simüle edilmedi.
Tehlike kontrolü bu karşılaştırmada yeniden kurulmadı; ara karede çalışmaya devam ettiği regresyon
testinde doğrulandı. Temas tahmininin doğrusal ara yoldan etkilenmesi ve ağır planlayıcının bekletmesi
devam eden sınırlamalardır.

## Doğrulama

114 ilgili test geçti:

- `test_commit`, `test_async_plan`, `test_planner`, `test_planner_modes`, `test_route_trials`,
  `test_key_pickup`, `test_viewer`: 106 test. İlk çalıştırmada 105 geçti, yerel UDP testi kısıtlı ortamda
  paket alamadı; aynı test sandbox dışında tekrar çalıştırılınca geçti.
- `test_analyze` ve `test_agent::test_runner_plays_a_recorded_wave`: 8 test.

Testler; kayıttan alınan zıplama örneğini, bir sonraki model adımında kalıcı sapmanın tespitini,
ara karede temas/tehlike/anahtar/hedef kontrollerini, asenkron plan aktarımını, gerçek rota için
bir-karelik denetimin korunmasını ve raporda interpolasyonun model hatası diye sayılmamasını kapsıyor.

Tek kayıt denetimi: 34,88 s duvar süresi, 34,72 s CPU, 403,5 MiB en yüksek süreç çalışma kümesi;
alt süreç yok. Süreler import sonrası tek çalıştırmaya ait, soğuk/sıcak planlayıcı performans ölçümü değil.
Token kullanımı ölçülmedi. Canlı oyun kontrol edilmedi.

Tekrar üretim: `python analysis/interpolation_20261116/audit.py` (smbrl dizininden).
Canlı kontrol için önceki çalıştırmanın argümanlarıyla, ayrı bir run adı altında 4-4 kullanılmalı;
`game_speed`, plan iptalleri, temas tetikleri ve bitirme oranı birlikte karşılaştırılmalı.

Bir sonraki model geliştirmesinin odağı, geçişlerin zamanlaması: tek ileri geçişte dört ara durumun
öğretilmesi, duvar/zıplama/iniş örneklerinin dengelenmesi ve ayrı tutulmuş dizilerde çok-adımlı doğrulama.
