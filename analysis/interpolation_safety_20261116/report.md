# Erken tepki gerilemesi — 16 Kasım 2026

`online_check06` sonrasında, ilk optimizasyondaki **ara-kare konum sapmalarını susturma** kuralı geri alındı.
Azalan tetik sayısı, kontrol kalitesinin korunduğunun kanıtı değildi. Ara konumlar interpolasyon olsa bile,
sapma gerçek bir hareketin gerçekleşmediğini gösterebilir ve bir sonraki model adımını beklemek geç olabilir.

## Kayıttaki somut örnek

4-1, deneme 5, kare 88: seçili hareket `run-jump R hold 10`. Model karakterin yükseleceğini öngörürken
gerçek karakter düşmeye devam ediyor:

| Kare | Konum hatası | İlk optimizasyon | Düzeltilmiş kontrol |
|---|---:|---|---|
| 89 | 4,72 | Bekle | Hemen yeniden planla |
| 90 | 9,59 | Bekle | Hemen yeniden planla |
| 91 | 14,62 | Bekle | Hemen yeniden planla |
| 92 | 19,79 | Yeniden planlandı | Doğrudan model noktası kontrolü sürer |

Gerçek kayıt 96. karede `h7#3` ile ölümle bitiyor. Erken yeniden planlamanın bu ölümü önleyeceği
burada kanıtlanmadı; doğrulanan, gereken geri bildirimin üç kare geciktirilmiş olmasıdır.
Nominal yolun yakınındaki tehlikeleri denetleyerek ertelemeyi koşullamak da bu örneği kaçırdı:
nominal yol zaten gerçekleşmeyen zıplamaya dayanıyordu. Bu yaklaşım üretim kodunda tutulmadı.

## Son kod

- `Committer` konum sapmasını tekrar **her karede** denetler. `path_step` raporlama metaverisidir;
  kontrol döngüsündeki tepkileri susturmaz. Model hata raporunda doğrudan tahmin ve interpolasyon ayrımı korunur.
- Tehlike denetimi, kalan tahmini yolu **gözlenen konum farkıyla kaydırarak** kontrol eder. Bu bir yeni
  dinamik tahmini veya oynatılan yol değişikliği değildir; gerçek konumdan yeniden plan isteme kontrolüdür.
- `interval`, `move end`, `short`, `missed` tetikleri tehlike kontrolünü engellemez. Böylece zamanlanmış,
  arka planda işlenebilecek bir karar acil tehlike uyarısını gölgelemez. Test, `interval + danger` birleşiminin
  asenkron işi iptal edip karar istediğini doğrular.
- Eşikler, model ağırlıkları, hareket seçimi/risk cezaları ve kalıcı hafızalar değiştirilmedi.

## Doğrulama

`audit.py`: `online_check06` 4-1 deneme 5 ve 4-2 deneme 50; aynı kaydedilmiş eylemler ve planlar.
Önceki denemelerden `FutureWorld` referansı kurulur; tahminler yalnızca o ana kadar görülmüş dünyadan
üretilir. Bölüm başındaki hata tüpleri aynen kullanılır. Ölümü açıklayan yerel tehlikeler vardır;
ayrı `DeathMemory` bu denetimde yoktur. Kaynak/Girdi SHA-256 özetleri `report.json` içinde.

| Kural | Karşılaştırılan kontrol | Erken konum sapması tetiği |
|---|---:|---:|
| Optimizasyon öncesi | 133 | 6 |
| İlk, koşulsuz ara-kare ertelemesi | 133 | 0 |
| Son düzeltme | 133 | 6 |

Bunlar aynı kayıt üzerindeki kontrol sonuçlarıdır: yeni eylemler üretilmedi, alternatif yaşama/ölme
sonuçları simüle edilmedi. Canlı başarı ve FPS doğrulanmadı. Eski erken tepkilerin geri gelmesi nedeniyle
ilk optimizasyonun hız kazanımının bir kısmı kaybolabilir.

Son kodla **70 test geçti**: `test_commit`, `test_async_plan`, `test_safety`, `test_future_cache`,
`test_key_pickup`, `test_agent::test_runner_plays_a_recorded_wave`.
Testler kayıtlı başarısız zıplamayı, ara-kare/eksik-gelecek tepkisini, gözlenen konum nedeniyle ortaya
çıkan tehlikeyi, zamanlanmış kararla aynı karede tehlikeyi, asenkron iptali ve önceki güvenlik kurallarını kapsar.

Denetimde 133 kontrolün toplam duvar/CPU süresi: eski kural 0,743/0,750 s; koşulsuz erteleme
0,763/0,797 s; düzeltme 0,737/0,703 s. Tek geçiştir; gelecek oluşturma, model ve oyun maliyetleri dahil
değildir, FPS/hızlanma kanıtı değildir. Soğuk/sıcak ayrımı, tepe bellek ve token maliyeti ölçülmedi.

## Ayrı kalan tekrar döngüsü

4-2 deneme 50: son planın ölüm karesindeki konum tahmini yaklaşık 0,73 birim hatalı; yani bu örnekte
ölümü yalnızca kötü fizik tahminiyle açıklayamayız. Kare 44'te finaldeki 16 adayın tamamı veto edilmiş.
Önceki kararların da `safe_alternatives=0` olması nedeniyle, tekrar eden ölümlerde hafıza kaydı
`no decision of this wave had a safe alternative: nothing charged` diyor. Bu, aynı hareket dizisinin
yeniden seçilmesini engellemiyor. Adayların güvenli bir seçenek üretememesi ve başarısızlık hafızasının
bu durumdaki davranışı ayrı incelenmeli; bu değişiklik onların düzeldiğini iddia etmiyor.

Tekrar üretim: smbrl dizininde `python analysis/interpolation_safety_20261116/audit.py`.
Çalışan ajan kodu otomatik yenilemez; yeni sürüm için ajan yeniden başlatılmalı ve yeni bir run adı kullanılmalı.
