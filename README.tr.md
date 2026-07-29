# mlx-finetune-test

[English](README.md) | [Türkçe](README.tr.md)

Apple Silicon üzerinde [MLX](https://github.com/ml-explore/mlx) kullanarak yerel bir LLM'in
LoRA fine-tuning'ini uçtan uca deneyimlemek için hazırlanmış bir pipeline. Bu bir öğrenme
projesi; odak noktası veri kalitesi veya model performansı değil, fine-tuning + inference
pipeline'ının uçtan uca çalışır hale getirilmesi.

Proje iki aşamadan geçti: önce dummy (sahte) bir veri setiyle fine-tuning → evaluation →
serving zincirinin uçtan uca çalıştığı doğrulandı (aşağıda "Aşama 1"), ardından aynı pipeline
gerçek bir görev üzerinde test edildi — phishing email sınıflandırması (aşağıda "Aşama 2").

## Neden MLX

İlk olarak [Unsloth](https://github.com/unslothai/unsloth) değerlendirildi, ancak
CUDA/Triton'a bağımlı ve Apple Silicon'da native çalışmıyor. MLX, Apple'ın kendi array
framework'ü ve M-serisi çiplerde first-class destek sunuyor, bu yüzden MLX seçildi.

## Model

[`mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit`](https://huggingface.co/mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit)
— abliterated Qwen3-4B'nin 4-bit MLX çevirisi (2.26GB). Abliteration, modelin refuse
(reddetme) davranışını kaldırır; yeni bilgi veya yetenek eklemez.

Model ağırlıkları bu repoda tutulmuyor — ilk çalıştırmada indirilip
`~/.cache/huggingface/` altında cache'leniyor.

## Proje yapısı

```
scripts/    inference, veri hazırlama, evaluation ve pipeline script'leri
data/       eğitim/validasyon verisi (train.jsonl, valid.jsonl) — git'te takip edilmiyor
adapters/   LoRA adapter çıktısı — git'te takip edilmiyor
outputs/    loglar, deney çıktıları ve checkpoint yedekleri — git'te takip edilmiyor
```

`data/train.jsonl` ve `data/valid.jsonl` veri formatı: satır başına bir JSON objesi,
`{"prompt": "...", "completion": "..."}`.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Aşama 1: dummy veri — pipeline doğrulaması

Gerçek veriye geçmeden önce, fine-tuning → evaluation → serving zincirinin genel olarak
çalıştığını doğrulamak için dummy bir veri seti (25 train / 5 valid örnek) kullanıldı. Bu
veri seti, modeli her cevaba sabit bir `[SOC-TEST]` marker'ı eklemesi için eğitti — base
modelin sergilemediği bir davranış, böylece başarılı bir fine-tune sadece loss eğrisinde
değil çıktıda da görülebiliyor.

```bash
mlx_lm.lora --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --train --data ./data --adapter-path adapters --iters 200 --batch-size 2
```

Sonuçlar: val loss 200 iterasyon boyunca 6.589'dan 0.504'e düştü. Adapter, eğitim setinin
dışındaki prompt'lara karşı test edildi:

- **Diller arası**: Türkçe bir prompt, eğitim verisi tamamen İngilizce olmasına rağmen
  yine marker'ı ve doğru bir cevap üretti.
- **Domain kayması**: phishing ile ilgili bir prompt (eğitim konularıyla alakasız),
  marker'la birlikte doğru ve tutarlı bir cevap üretti — base modelin bilgisi korundu,
  adapter onu ezmedi.
- **Parafraz**: eğitim setindeki bir sorunun yeniden ifade edilmiş hali doğru cevaplandı,
  bu da basit ezberlemeyi eledi.
- **Serving sonrası**: fuse edilip GGUF'a çevrildikten ve Q4_K_M'e quantize edildikten
  sonra, llama.cpp'nin API'si üzerinden servis edilen cevaplarda marker hâlâ mevcuttu —
  tam MLX'ten GGUF'a dönüşüm zinciri, fine-tune edilmiş davranışı koruyor.

Bu aşama ayrıca dikkat çekici bir yan etki ortaya çıkardı: base model, basit prompt'larda
bile varsayılan olarak bir `<think>...</think>` reasoning bloğu üretiyor. Completion'larında
hiç reasoning izi olmayan bir veri setiyle fine-tune edildikten sonra, model bunları
tamamen üretmeyi bıraktı (boş `<think></think>`). Dar bir davranış kalıbı üzerinde
fine-tuning yapmak, ilgisiz varsayılan davranışları da bastırabiliyor.

## Aşama 2: gerçek veri — phishing email sınıflandırması

Pipeline doğrulandıktan sonra, dummy görev gerçek bir görevle değiştirildi: emailleri
`Phishing` veya `Safe` olarak sınıflandırmak, Kaggle
["Phishing Email Detection"](https://www.kaggle.com/datasets/subhajournal/phishingemails)
veri seti kullanılarak (18.650 email, `Email Text` + `Email Type` kolonları).

### Veri hazırlama

`scripts/prepare_data.py`, ham CSV'yi `{"prompt": ..., "completion": ...}` JSONL formatına
çeviriyor; temizleme adımları ham veride bulunan sorunlardan yola çıkıyor:

- **Boş satırlar** (19) ve **duplicate email text'ler** (1.068) siliniyor — duplicate'ler
  önemli çünkü temizlenmezlerse aynı email hem train hem valid'e sızabilir (data leakage).
- `--max-length` (varsayılan 1000 karakter) üstü satırlar atılıyor. Ham verinin uzunluk
  kuyruğu medyan 880 karaktere karşı 17M karaktere kadar çıkıyor; daha önce denenen 5000
  karakter sınırı fine-tuning'i pratik olmayacak kadar yavaşlatmıştı.
- Train/valid split'i (%90/%10) etikete göre **stratified** yapılıyor, böylece her iki set
  de kaynak verideki Safe/Phishing oranını (~%61/%39) koruyor.

```bash
python scripts/prepare_data.py --max-length 1000
```

### Fine-tuning

```bash
mlx_lm.lora --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --train --data ./data --adapter-path adapters --iters 1000 --batch-size 4
```

### Val loss'a fazla güvenmenin verdiği ders

Val loss monoton bir şekilde azalmadı: 4.761 (başlangıç) → 3.033 (iter 200) →
3.223 (iter 400) → 3.190 (iter 600) → 2.955 (iter 700) → 3.151 (iter 1000).
Naif okuma — "iter 200 en düşük loss'a sahip, onu kullan" — yanlış çıktı: iter 200
checkpoint'i elle test edildiğinde her email için `Safe` tahmin ediyordu, açık phishing
örnekleri dahil. Model, gerçek ayrımı değil, çoğunluk sınıfını (Safe verinin ~%61'i)
öğrenmişti — bir loss eğrisi, model hiçbir işe yarar şey öğrenmemişken bile iyi
görünebiliyor.

Eğitimin daha ileri aşamalarındaki checkpoint'ler (iter 700 ve iter 1000), elle yazılmış
6 email'e (3 phishing, 3 safe, hiçbiri eğitim setinden kopyalanmadı) karşı test edildi ve
ikisi de 6/6 skor aldı — eğitim sırasında hiç görülmemiş içeriği doğru sınıflandırdı.
**Ders: checkpoint seçimini sadece val loss'a dayandırma — karara varmadan önce gerçek
generation çıktısıyla doğrula.** Iter 1000, nihai adapter olarak tutuldu.

## Evaluation

`scripts/evaluate.py`, fine-tune edilmiş adapter'ı yukarıda bahsedilen elle yazılmış 6
phishing/safe email'e karşı çalıştırır ve tahmin edilen etiketi beklenenle karşılaştırır:

```bash
python scripts/evaluate.py --adapter-path adapters
```

Bir pass/fail özeti basar (mevcut adapter'da 6/6) ve tam completion'ları
`outputs/eval_results.json`'a yazar.

## Serving

MLX yalnızca Apple Silicon'da çalışıyor, bu yüzden tek başına production serving hedefi
değil. Fine-tune edilmiş modeli standart bir sunucuda (örn. Linux) çalıştırmak için, LoRA
adapter base modele fuse edilip [llama.cpp](https://github.com/ggml-org/llama.cpp) ile
kullanılmak üzere GGUF'a çevriliyor. Bu zincir hem Aşama 1 (dummy) hem de Aşama 2
(phishing) adapter'larıyla uçtan uca doğrulandı.

`scripts/run_pipeline.py`, fuse → GGUF dönüşümü → quantization → (isteğe bağlı) serving
adımlarını zincirliyor:

```bash
export LLAMA_API_KEY="bir-anahtar-seç"  # opsiyonel ama önerilir, aşağıya bakın
python scripts/run_pipeline.py --serve
```

`--llama-cpp-path`, `LLAMA_CPP_PATH` env var'ı veya `~/llama.cpp` üzerinden (bu sırayla)
bir llama.cpp checkout'u arar. Hiçbiri bulunamazsa, hata vermek yerine MLX fuse adımından
sonra durur — llama.cpp pip ile kurulmuyor, bu yüzden konumu makineye göre değişebiliyor.
Tüm seçenekler için (`--adapter-path`, `--quant-type`, `--port`, ...)
`python scripts/run_pipeline.py --help` çalıştırın.

**Auth ve CORS**: `llama-server` varsayılan olarak API key kullanmıyor ve
`--cors-origins '*'` (herhangi bir web sitesi tarayıcıdan çağırabilir) ile çalışıyor.
`run_pipeline.py`, CORS varsayılanını `localhost`'a daraltıyor ve `LLAMA_API_KEY` env
var'ı set edilmişse `llama-server`'a `--api-key` geçiyor — auth'suz çalıştırmak için
(örn. local test) boş bırakın, ya da her istekte bearer token zorunlu kılmak için set
edin. Daha permissive varsayılana dönmek için `--cors-origins '*'`'i açıkça geçin.

<details>
<summary>Karşılık gelen manuel komutlar</summary>

**1. Adapter'ı base modele fuse et (dequantized, fp16):**
```bash
mlx_lm.fuse --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --adapter-path adapters --save-path fused_model --dequantize
```

**2. GGUF'a çevir.** `mlx_lm.fuse --export-gguf`, Qwen3 mimarisini desteklemiyor, bu
yüzden dönüşüm bunun yerine llama.cpp'nin kendi script'inden geçiyor:
```bash
python ~/llama.cpp/convert_hf_to_gguf.py fused_model \
  --outfile fused_model/model.gguf --outtype f16
```

**3. Quantize et** (fp16 → Q4_K_M, ~8GB → ~2.4GB):
```bash
~/llama.cpp/build/bin/llama-quantize fused_model/model.gguf \
  fused_model/model-q4_k_m.gguf Q4_K_M
```

**4. OpenAI-uyumlu bir API servis et:**
```bash
~/llama.cpp/build/bin/llama-server -m fused_model/model-q4_k_m.gguf --port 8080
```
</details>

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLAMA_API_KEY" \
  -d '{"messages": [{"role": "user", "content": "Your question here"}], "max_tokens": 100}'
```

## Durum

- [x] mlx-lm kuruldu, baseline inference test edildi
- [x] Aşama 1: dummy `[SOC-TEST]` verisiyle LoRA fine-tuning, tam pipeline doğrulandı
      (fine-tune → evaluate → fuse → GGUF → quantize → llama.cpp server → API)
- [x] Aşama 2: gerçek veri seti (Kaggle phishing email) seçildi, temizlendi ve LoRA
      fine-tuning için kullanıldı; checkpoint seçimi sadece val loss'a değil gerçek
      çıktı doğrulamasına dayandırıldı
- [x] Serving zinciri (fuse → GGUF → quantize → llama.cpp) Aşama 2 adapter'ıyla yeniden
      çalıştırıldı ve doğrulandı — API, phishing/safe test emaillerini doğru sınıflandırdı
- [x] Serving konfigürasyonu: opsiyonel API key auth (`LLAMA_API_KEY`), CORS varsayılan
      olarak `localhost`'a kısıtlı
