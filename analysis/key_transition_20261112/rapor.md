# Hayal edilen anahtar geçişi ve 2-10 kararı — 12 Kasım 2026

Anahtar edinimi normal planner'ın simülasyonuna bağlandı. Her aday ve ensemble üyesinin maskesi ayrı ilerliyor; sonraki karo girdisi, kilit çarpışma kontrolü ve destek hesabı o maskeyi kullanıyor. Mevcut 12 epoch modelinin ağırlıkları değiştirilmedi, yeni eğitim yapılmadı. Oyun veya save dosyalarına bağlanılmadı.

## Uygulanan kural

`game.level.LockContext` anahtar konumları ve kilit gruplarının yanında deterministik edinim kuralının da sahibi. Yerel `realgame.js` kuralına göre fizik güncellemesinden SONRA konumun anahtara uzaklığı kesin olarak 22'den küçükse anahtar ediniliyor. Anahtarlar rota sırasından bağımsız toplanıyor; ilerleme etabı sahip olunan maskeden çıkarılıyor.

Gelen model adımının çarpışma kontrolü, adım başındaki maske ile yapılıyor. Sonra edinim hesaplanıyor; açılan kilit bir SONRAKİ model adımında kullanılabiliyor. Bir karelik modelde bu uç nokta sırası native kuralla eşleşir. Dört karelik modelde aradaki dört tam kare konumu doğrusal kestiriliyor: sürekli segmentin anahtara yaklaşması tek başına yetmez. Bu ara yol fizik tahmini değildir. Yeni kilit durumunun kullanılmasında, kendi kestirilmiş edinim anına göre en fazla üç kare gecikme olabilir; gerçek kayda göre zaman hatasının üç kareyle sınırlı olduğu iddia edilmez.

`wm.rollout` için yeni davranış açık `key_pickup=True` seçimi ister; varsayılan kapalıdır. Normal `Planner` bunu açar. Eski deneylerin varsayılan ham rollout sözleşmesi ve checkpoint özellik boyutları korunmuştur. `Committer` gerçek ve tahmin edilen destek durumunu özel maskeyle kontrol eder; anahtarlı kilidi kapalı sanarak destek değişimi üretmez.

Kayıtlı en iyi rotada ayrıca gerçek maskeler tutuluyor. Anahtarlı haritada bilinmeyen veya farklı başlangıç maskesi, kayıtlı yolu güvenilir sayan istisnayı kullanamaz; aday model ve guard ile değerlendirilir. Bilinen son maske destek ve kalan süre hesabına gider. İlk can değerlendirmesinde bu rota belleği zaten yoktur.

## Gerçek kayıt doğrulaması

Kaynak, sabit `tile_shapes_epochs_v1/dataset.json` manifestinin 2-10 doğrulama ajanlarıdır. Eğitim örnekleri bu karşılaştırmaya alınmadı. Model SHA-256: `d979f1dd1b42bd36488139496a3a0f4a7c269de595388d318a686ca2c41b2673`.

- Ortak yeni kural, gerçek konumlar üzerinden 38.032 ardışık geçiş ve 45 anahtar ediniminde kayıtlı maskeyi **sıfır farkla** yeniden üretti. Bu, kayıtlı konumdan maskeye giden kural testidir; modelin hareket tahmini değildir.
- Değerlendirmeden önce sekiz anahtar alma ve sekiz anahtarsız yakın geçiş penceresi sabitlendi. Her pencere 48 kare; aynı durumlar ve gerçek eylemler, edinim kapalı/açık olarak, hem ham rollout hem planner guard ile karşılaştırıldı.
- Beş üyenin çoğunluğu ölçüldüğünde yeni ham ve guard'lı simülasyon **8/8** gerçek edinimi tahmin etti. Kapalı eski davranışta maskeler sabit kaldığı için edinim **0/8** idi. Sekiz negatif örneğin **birinde yanlış edinim** öngörüldü; beş üye de bu örnekte anahtarı aldığını düşündü. Bu, ensemble uzlaşmasının tek başına doğruluk garantisi olmadığını gösterir. Tahmin edilen konum/ara yol gerçek yoldan farklılaşabilir; bu denetim fizik model hatası ile doğrusal ara yol hatasını bütünüyle ayırmaz.
- Edinim saptanan pozitiflerde ilk anahtarlı dört-kare sınırının gerçek edinim karesine göre imzalı zaman hatası ortancası 0, p90 değeri +4 karedir. Küçük hedefli örneklem bir güvenilirlik oranı değildir.
- Bu pencerelerde 48-kare hareket hatası azalmadı: guard'lı konum ortanca/p90 hatası her iki durumda yaklaşık **14,66 / 107,56** oyun birimidir. Anahtar durumunun tutarlılığı kazandırıldı; uzun hareket tahmin hatası sürüyor.

Anahtar ediniminden kilit bölgesine gerçek ulaşım bu doğrulama kayıtlarında en az **260 kare** sürüyor. Bu yüzden 48 karelik model tahminini bütün geçişin kanıtı saymadık. Sekiz daha sonraki kilit penceresi ayrıca sabitlendi:

- Anahtar öncesinden kilide kadar GERÇEK konumlar üzerinden maske taşındı; sekiz kesitte kayıtla fark yok. Bu öğretmen zorlamalı kural replay'idir, 260+ karelik açık çevrim dünya tahmini değildir.
- Aynı kayıtlı kilit giriş hareketine guard'da maske0 verilince sekizinin hepsi engellendi; maske1 verilince sekizi de engellenmedi. Bu özel kilit geometrisinin kontrolüdür, canlı oyunda anahtarsız ajanla çarpışma deneyi değildir.
- Daha sonraki model pencereleri anahtar zaten eldeyken başlar. Bunlar bilinen maske tutarlılığına bakar; hayalde anahtar alıp 260 kare sonra kilide varmayı doğrulamaz.

## 2-10'daki takılma kararı

Eski kayıt `collect_2-10_fan_keys_01`, wave_00001, ajan0, kare200'deki aynı gerçek durumdan iki taze planner kararı üretildi. Anılar ve kayıtlı rota kullanılmadı; gelecekteki kaydedilmiş tehlike/platform bilgileri verildi. Kayıttaki ilk harita anından yapılan rekonstrüksiyonun eski başlangıç loguyla küçük farkı önceki alan raporunda açıklanmıştır.

| Model | Seçilen ilk hareketler | Başlangıç T | Son T | Değerlendirilen kare |
| --- | --- | ---: | ---: | ---: |
| Eski wm_step4 | Sola kısa zıpla, bekle | 317,38 | 318,38 | 48 |
| Yeni 12 epoch | Sağa uzun zıpla, sağa uzun zıpla | 317,38 | 295,74 | 60 (48 + 12 devam) |

Eski modelin son adaylarında sağa giden ilk hareket kalmamışken, yeni modelde sağa uzun zıplama ilk aşamada güçlü aday oluyor ve son seçimde kazanıyor. Yeni seçilen 48-kare yolun en sağ noktası yaklaşık x239; eski seçimin yaklaşık x31. Anahtar değişikliği öncesi ve sonrası bu iki kararın hareket/puanları aynı kaldı. Dolayısıyla bu tek durumda yeni model zaten yararlı aday üretiyor; ilerleme alanını veya ufku bu aşamada değiştirmedik.

Bu tek karar canlı oyunda takılmanın bittiği veya bölümün tamamlanabildiği kanıtı değildir. Alanın iyimser düzlüğü önceki rapordaki gibi duruyor; bunun gerekli bir sonraki düzeltme olduğuna canlı gözlem olmadan karar vermiyoruz. Eski/yeni model karşılaştırmasında veri ve girdi gösterimi de farklıdır; farkın tek nedenini izole etmedik.

## Kontroller ve sonraki adım

18 odaklı anahtar testi geçti; ilgili feature, experiment ve commit regresyonları da çalıştırıldı. Son gerçek model denetimi 24 pencerede 6,88 saniye sürdü. Eski dondurulmuş 12 epoch değerlendirmesi yeni kodda `key_pickup=False` ile yeniden koşuldu: 272.085 örneğin kısa tahmin panelleri ve 128 başlangıcın ileri tahmin panelleri **tam olarak aynı** çıktı. Eski rapor değiştirilmedi; karşılaştırma `frozen_rollout_compatibility.json` içinde.

Bir sonraki doğrulama, kullanıcı oyunda 2-10'u seçtikten sonra yeni 12 epoch modeliyle tek-can işlev kontrolüdür. Bu bölüm eğitim verisindedir; görülmemiş bölüm/genelleme testi sayılmaz. Kalıcı anı veya tekrar dalgalarıyla sonucu iyileştirmeden planlar, gerçek anahtar maskesi ve ölüm/bitiş incelenmeli.

Ham kanıt: `cases.json`, `gate_cases.json`, `evaluation_final.json`, `baseline_stall.json`, `after_stall.json`, `frozen_rollout_compatibility.json`. Önceki ara değerlendirme `evaluation.json` da korunmuştur. Bütün kaynak kayıtlar, manifestler ve model dosyaları korunmuştur; commit yapılmadı.
