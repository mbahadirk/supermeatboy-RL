# SMB-RL Gemini Changes & Architecture Documentation

Bu dosya, projede yapılan değişiklikleri, eklenen modülleri, mimari kararları ve diğer coding agent'lar için teknik detayları içerir.

---

## 1. Repo Ayrıştırması ve Durum Kaydı (Git State)
- **Önceki Durumun Kaydedilmesi:** `smbrl` içerisindeki tüm yeni dosyalar (`ingame.py`, `ingame_arena.js`, `realgame.py`, `realgame.js`, `tools/smbl_extract.py`, input probları ve simülatör güncellemeleri) Git commit `22a3b6b62` ile `feat(smbrl): in-game multi-agent arena, realgame frida hooks, and training updates` başlığı altında başarıyla commit edildi.
- **Morluto Upstream Ayrıştırması:** Morluto'nun projesinden bağımsız bir repo haline getirmek için eski `origin` (`https://github.com/morluto/rea.git`) kaldırıldı (`git remote remove origin`). Kullanıcı kendi GitHub/GitLab remote URL'sini ekleyebilir (`git remote add origin <url>`).

---

## 2. Dijkstra Tabanlı "Reward Rail" Mekanizması (`reward_rail.py`)
Mevcut ortamda kullanılan 4-bağlantılı basit BFS veya Euclidean mesafe hesabı, platform oyunlarındaki yerçekimini ve duvarları hesaba katmadığı için ajanların hedef duvarın arkasındayken zıplayarak yerel minimumlara takılmasına (local optima trap) neden oluyordu.

### Çözüm ve Algoritma:
- **`RewardRail` Modülü Oluşturuldu:**
  1. **Platformer-Aware Dijkstra Arama Grafı:**
     - Harita hücreleri (W x H) analiz edilir: Solid bloklar (`#`), ölümcül engeller (`^`, testereler, lavlar), hareketli engeller (`movers`), warp portalları, crumble blokları, anahtarlar/bandajlar ve hedef (Princess / Bandage Girl).
     - **Tehlike Güvenlik Tamponu (Hazard Buffer):** Testere ve sivri uçların 1-2 blok çevresindeki açık hücrelere yüksek risk maliyeti eklenerek, bulunan yolun ölümcül nesnelerden güvenli bir koridorda geçmesi sağlandı.
     - **Yerçekimi ve Platformer Ağırlıkları:**
       - Yerde yatay yürüme/koşma maliyeti: Düşük (`1.0 - 1.8`).
       - Yerçekimiyle aşağı düşme: Kolay (`0.7`).
       - Zeminden veya duvardan yukarı zıplama: Normal (`1.1 - 1.6`).
       - Havada desteksiz yukarı hareket: Yüksek ceza (`2.0 - 3.5`).
       - Portallar: 1 blok maliyetle anında teleport.
  2. **Çoklu Hedef (Multi-Goal Routing):**
     - Haritada anahtar veya bandaj (`level.meta["bandages"]`) varsa, yol `Spawn -> Anahtar(lar) -> Prenses` rotası şeklinde Dijkstra ile hesaplanır.
  3. **Yol Parametrelendirmesi & İzdüşüm (Arc-Length Parameterization):**
     - Dijkstra düğümleri dünya koordinatlarına (piksel) çevrilir ve kümülatif mesafe hesaplanır.
     - Herhangi bir $(x, y)$ noktasının bu rail üzerindeki en yakın izdüşümü (ilerleme mesafesi $s$, raydan dikey sapma $d_\perp$, en yakın segment) analitik olarak hesaplanır.

---

## 3. Ödül, Ceza ve Fitness Formülasyonu (`env.py` & `realgame.py`)
Eski zayıf ödül sistemi yerine ray odaklı ve stagnation-önleyici yeni mekanizma getirildi:
- **Reward Rail İlerleme Ödülü (`W_RAIL_PROG`):**
  - Ajan ray üzerinde daha önce ulaştığı en yüksek noktayı (`_rail_best_dist`) aştığı her piksel için pozitif ödül alır (`+0.08 * pixel`).
- **Yol Yakınlık Şekillendirmesi (Soft Tether):**
  - Ajanın raydan çok fazla uzaklaşıp çıkmaz sokaklara girmesini engellemek için hafif bir sapma cezası uygulanır (`-0.005 * min(d_perp / TILE, 6.0)`).
- **Prenses / Bitiş Ödülü:**
  - Seviyeyi tamamlayıp prensese ulaşıldığında: `+50.0` büyük ödül + zaman verimliliği bonusu (hızlı bitirişe göre `+20.0`'ye kadar ek bonus).
- **Anahtar / Bandaj Ödülü:**
  - Haritadaki anahtarlara veya bandajlara ulaşıldığında: `+25.0` ek bonus.
- **Ölüm Cezası:**
  - Ölüm anında: `-5.0` ceza.
- **Stagnation (Durgunluk) ve Eylem Tekrarı Cezası:**
  - Ajan 40 frame boyunca aynı 2 piksellik alanda zıplıyorsa / duvara takılıyorsa: `-0.02` per frame.
  - 150 frame (2.5 saniye) boyunca ilerleme kaydedemezse: `-0.08` ceza ve bölüm erken sonlandırma (`truncated = True`).
  - Duvara doğru sürekli aynı girdiyi basma cezası: `-0.015`.
- **Fitness Metriği:**
  $$\text{Fitness} = 1000 \cdot \mathbb{I}(\text{won}) + \left(\frac{s_{\max}}{L} \times 500\right) + (\text{keys} \times 100) - (\text{steps} \times 0.05) - (\text{deaths} \times 50)$$
  Bu sayede modellerin nesiller boyu performansı sayısal ve net olarak kıyaslanabilir.

---

## 4. Gelişmiş RL & Sinir Ağı Telemetri Arayüzü (`viewer.py`)
Mevcut arayüz yetersiz olduğu için `viewer.py` baştan sona modernize edildi:
- **Yarı Saydam Yeşil Reward Rail Çizimi:**
  - Spawn'dan Prenses'e kadar olan optimal yol, harita üzerine alfa kanallı (`RGBA: 45, 235, 95, 150`) kalın yeşil çizgi ve parlayan waypoint noktalarıyla çizildi.
  - Anahtarlar sarı elmas, Prenses ise parlayan halka ile vurgulandı.
- **Canlı Sinir Ağı (PPO Policy) Paneli:**
  - Sağ panele sinir ağının o anki durumunu gösteren telemetri eklendi:
    - **Aksiyon Olasılıkları Dağılımı ($P(a|s)$):** 10 discrete aksiyonun (`NOOP`, `LEFT`, `RIGHT`, `JUMP`, `L+JMP`, `R+JMP`, `R+RUN`, `R+R+J`, `L+RUN`, `L+R+J`) anlık softmax olasılıkları renkli bar grafiğiyle canlı gösterilir. Ajanın o an seçtiği aksiyon parlak yeşil renkle vurgulanır.
    - **Değer Fonksiyonu ($V(s)$):** Critic ağının anlık durum için tahmin ettiği beklenen toplam getiri.
    - **Politika Entropisi (Entropy):** Ajanın ne kadar keşif (exploration) yaptığını veya ne kadar kendinden emin (exploitation) olduğunu gösterir.
    - **Ray İlerleme Göstergesi (Progress Bar):** Rayın yüzde kaçının tamamlandığı (`[████████░░░░] 48.2%`).
    - **Fitness & Stagnation Sayacı:** Ajanın o anki fitness skoru ve takılma sayacı.
- **Klavye Kısayolları:**
  - `[TAB]`: Sinir ağı telemetri panelini açıp/kapatır.
  - `[D]`: Yeşil Reward Rail çizimini açıp/kapatır.
  - `[S]`: Gerçek zamanlı (60 Hz) ve hızlandırılmış mod arasında geçiş yapar.
  - `[V]`: Çizimi tamamen açıp/kapatarak maksimum eğitim hızına çıkar.

---

## 5. Gerçek Oyun ve Frida Hook Güncellemeleri (`ingame.py` & `ingame_arena.js`)
- `ingame.py` içerisindeki `_rail_to_game` fonksiyonu, Reward Rail koordinatlarını Super Meat Boy oyun içi koordinat sistemine dönüştürerek Frida scriptine iletir.
- `ingame_arena.js` içerisinde DRAW hook'una eklenen rutin ile ray waypoint'leri oyun içerisinde yarı saydam Meat Boy silüetleri/işaretçileri olarak çizilir.

---

## 6. Diğer Coding Agent'lar İçin Notlar ve Çalıştırma Rehberi
1. **Model Eğitimi Başlatmak (Görsel Arayüz ve Ray İle):**
   ```bash
   python smbrl/train.py --level smb:1-1 --n-envs 8 --watch --save-best
   ```
2. **Yerleşik Haritalarda Test:**
   ```bash
   python smbrl/train.py --level gap --n-envs 8 --watch
   ```
3. **Kayıtlı Modeli İzlemek:**
   ```bash
   python smbrl/watch.py runs/<run_folder>
   ```
4. **Kod Değişiklikleri Yaparken:**
   - Yeni bir seviye eklendiğinde `RewardRail` sınıfı seviyeyi otomatik olarak analiz eder ve en kısa güvenli yolu Dijkstra ile çıkarır.
   - Ödül parametreleri `env.py` ve `reward_rail.py` içerisindeki katsayılardan ayarlanabilir.

