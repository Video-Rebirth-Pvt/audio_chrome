# Chrome-10B：32 kHz 立体声 T2A 生成

纯音频生成模型，先做文本生成音频（T2A），视频放到最后一个阶段。音频链路基本复刻 MiniMax-H3：同一套 Audio VAE、同一种单流 packed DiT、同一套 rectified flow。本工程链路说明在 `docs/chrome_pipeline.html`。H3 的分析笔记在 `docs/minimax_h3_analysis.html`。

`MiniMax-H3`、`DiffSynth-Studio` 和 `VR-Foley-5.5B` 只读，作为参考实现和数据来源，不在里面改代码。本工程的代码、配置和脚本都放在 `Chrome-10B/` 下。

当前进度：S0 的 codec 复刻和 H3 权重对齐已完成；S1 数据管线已完成，VR-Foley 的五个数据源已经建好索引。S2 的 T2A 生成链路已经搭好并在 500 条上跑通（训练 / 验证 / 续训 / preview / 推理 / 导出），DDP、FSDP2、HSDP 三种并行都验证过，10B 在单机 8 卡 FSDP 上能训。下一步是 1B 消融。

## 已定的决定

| 项 | 决定 |
|---|---|
| VAE 路线 | 冻结 H3 的 Audio VAE，不重训。DiT 直接在它的白化 latent 上训练，白化统计量沿用 H3 的 |
| 文本编码器 | Qwen3-VL 系列，和 H3 一致。本地是 H3 的 Qwen3-VL-32B 完整权重（64 层 + 视觉塔），加载时只建前 50 层；最终用多大规模，在 1B 消融里定 |
| 任务 | 前期只训 T2A；结构上保留视频位置（AdaLN 三档、MM-RoPE 的 h/w 轴），S5 加视频时不改骨架 |

## 阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| S0 基础设施 | codec 复刻，H3 权重严格加载，decode/encode 对齐，导出往返 | 已完成 |
| S1 数据管线 | 清单 → 每个源一份 Arrow 索引，整秒长度策略，多源加权混合，按 token 预算打包，可续训；VR-Foley 五个源已建索引 | 已完成 |
| S2 T2A 生成链路 | 文本编码器 + 缓存、DiT（packing、MM-RoPE、AdaLN）、flow、task / trainer（DDP / FSDP2 / HSDP）、推理、导出；500 条上端到端已跑通。接下来 1B 消融，定文本编码器规模、shift、数据配比、AdaLN 形式、文本归一化 | 框架完成，消融未开始 |
| S3 10B 预训练 | 用 S2 定下的配方 | 规划 |
| S4 后训练 | CFG 蒸馏，少步蒸馏 | 规划 |
| S5 视频 | 接 VisualVAE，同一序列联合去噪 | 规划 |

## 音频约定

改动只在 `configs/_base_/audio_spec.yaml` 一处做，codec、数据和 DiT 的配置都继承它。代码里的换算统一走 `chrome/spec.py` 的 `AudioSpec`。

| 项 | 值 |
|---|---|
| 波形 | 32 kHz 立体声，`[C=2, N]` float32 |
| hop | 800 个采样点，即每秒 40 帧 latent（每个声道各一套） |
| latent | 每声道 32 通道，`[..., C, 32, T]` |
| 时长 | 4–15 秒；1 秒正好 40 帧，按整秒分桶不需要取整 |
| DiT 行数 | 每个声道每帧占一行：`rows = frames × 2`，15 秒是 600 帧、1200 行 |

单声道复制成左右相同；多于两声道的，额外声道按 0.7071 的权重折进左右（`chrome/data/audio_io.py` 的 `to_stereo`，逻辑同 VR-Foley）。训练读取时峰值超过 0.95 才整体缩小，更小的不放大，和 VR-Foley 相同（`data.peak`）；`load_audio` 默认不动峰值（`peak=None`）。

## 编解码器（已实现）

`chrome/codec/` 的 `ChromeAudioVAE` 复刻 H3 Audio VAE，state dict 的键名和 H3 一致，原始权重用 `strict=True` 直接加载，不需要改名钩子。

结构：DAC 编码器 → 因果注意力瓶颈 → 对角高斯 → BigVGAN 解码器。网络本身是单声道（`in_channels=1`），立体声折进 batch 维，左右声道共享全部权重，codec 内部不混合左右。

| 部分 | 结构 | 参数 |
|---|---|---|
| encoder | Snake 激活，下采样率 `[2,4,4,5,5]`，宽度 64 → 2048 | 73.72M |
| pre_block | `AttnProjection`：8 头因果注意力，多头输出取平均，再从 256 池化到 32；外加 GeGLU | 12.67M |
| heads | `mean_proj` / `logs_proj` 都是 `Conv1d(32,32,1)`，`dec_in_proj` 为 32 → 2048 | 0.07M |
| decoder | BigVGAN，上采样率 `[5,5,2,2,2,2,2]`、核 `[9,9,4,4,4,4,4]`、初始 1024 通道；AMP 残差块核 3/7/11、膨胀 1/3/5；抗混叠 SnakeBeta（logscale）；末端 clamp 到 [-1,1]，不用 tanh，`conv_post` 无 bias | 64.87M |
| 合计 | | 151.3M |

接口：

- `encode_audio(wave [..., C, N])` → 白化后的后验均值 `[..., C, 32, ceil(N/800)]`，float32，不带梯度。右侧补零到 800 的整数倍。
- `decode_audio(latent [..., C, 32, T])` → `[..., C, T·800]`。
- 训练用 `forward(audio, sample=True)`，返回 `(recon, posterior)`；`encode_posterior` 返回 `DiagonalGaussian`，其中 `logs` 是 log σ。
- DiT 只吃后验均值，不采样。

权重归一化用老式的 `weight_g` / `weight_v` 两个参数，前向时用 `torch._weight_norm` 合成权重。官方的 `init_weights` 对参数化后的 weight norm 卷积实际不起作用；这里按它 docstring 的本意实现：v 取截断正态（std 0.02），g 取 1。

白化：`(μ − mean) / std`，每个通道一对，左右声道共用同一对。数值放在 `LatentStats` 里，不进 state dict，模型转 bf16 时也不会丢精度。

权重格式（`chrome/codec/h3_io.py`）：

- **bundle**：`FL2VA/audio_vae`，由 `config.json` 指向 `config.yaml`、`metadata.json` 和 `model.safetensors`。metadata 里没有解码器核大小，按采样率查表：32000 对应 `(9,9,4,4,4,4,4)`。
- **diffusers**：根目录的 `audio_vae`，一个 `config.json` 加 `diffusion_pytorch_model.safetensors`。和 bundle 的权重逐位相同。导出一律写这个格式，额外记 `attn_proj`、`snake_logscale`、`use_tanh_at_final`、`use_bias_at_final`。
- 两种格式都把 `latents_mean` / `latents_std` 写在 `config.json` 里。

### 在 8 条 preview 音频上实测（H3 原权重，每条 10 秒）

- 白化后每个通道的 std 在 0.42–0.56 之间，不是 1。H3 给的 `latents_std` 大约是这批音频实测值的 2 倍，DiT 看到的数据幅度只有噪声的一半左右，相当于把信噪比调度整体挪了一截。VAE 冻结、统计量沿用 H3，这个偏移在 S2 调 shift 时一起看。
- 后验接近确定性：σ 中位数约 0.005，每维 KL 约 5.3。H3 VAE 实际接近普通自编码器，KL 权重很小。
- 人声的高频损失明显：vocal lyric 在 8–16 kHz 只保留了 29% 的能量，其他音频在 57–80%。
- mel_l1 在 0.15–0.32 之间（中文女声最高）。SI-SDR 差异很大：rain −19.6 dB，rock drums 17.7 dB。

## H3 对齐检查

`tests/test_h3_parity.py`，入口 `scripts/vae/check_h3_parity.sh`。检查时关闭 TF32，容差 1e-4。共 14 项：

1. 两种格式都能严格加载，配置等于 `configs/vae/model_h3.yaml`，并且符合音频约定。
2. 随机 latent 下 decode 对比官方 bundle：1.7e-6。
3. encode 的 mean / logs 对比官方子模块：7.5e-5 / 5.2e-6；encode、白化后的 `encode_audio`、`decode_audio` 对比 DiffSynth：9.6e-5 / 5.7e-5 / 2.8e-7。
4. 左右声道相同则 latent 相同；批量编码等于逐条编码，误差为 0。
5. 导出再加载，逐位一致。
6. 真实音频：用 `mel_l1 < 0.5` 作门槛；SI-SDR、σ、KL、白化统计只写进报告。

真实音频不用 SI-SDR 作门槛，原因是 GAN 解码器重建雨声这类噪声纹理时频谱对、相位是新生成的，官方模型自己在 rain 上就是 −19.57 dB。结果写在 `outputs/vae/parity/report.json`，每条音频另存 `<label>_src.wav` 和 `<label>_rec.wav`，可以直接试听对比。

## DiT（已实现，`chrome/dit/`）

照搬 H3 的约定，只是把规模缩小。参数名和 DiffSynth 的 `MiniMaxH3DiT` 一致：

- 单流 packed varlen 序列，一个包里放多个样本。模态标签：VIDEO=0、TEXT=1、AUDIO=2、PAD=-1。
- AdaLN 分三档模态，按 `unique_t × 3 + tag` 取值，只在不同的 t 上计算。纯音频阶段只用到 TEXT 和 AUDIO，但保留三档。
- MM-RoPE 覆盖 (t, h, w) 三轴，每轴 16 个频率，在 head_dim 128 的前 96 维上做 split-half 旋转。音频第 k 帧的 t 是 `text_len + k`；左右耳靠 w 区分，钉在画布左右两端（`_INTERP=32`），T2A 暂用 16:9 默认画布（`w_L≈−5.17`、`w_R≈36.16`）。
- 文本先过 `condition_proj` 投到 hidden，再过 2 层 token refiner；MLP 用 SwiGLU；Q/K 做 RMSNorm。
- 文本行的 AdaLN 时间步跟音频（`t_audio`），S5 加视频后也不改。H3 里文本行用的是 `t_video`，T2A 没有视频，所以这里不跟 H3。
- 音频每行的输入、输出都是 32 维，即一个声道一帧。

**待定风险：左右耳的 w 绑在视频画幅上。** H3 的 `_audio_w_axis` 取视频网格 `w_grid` 的首尾两列，画幅一变，两耳的 w 和间距都变：16:9 是 −5.17 / 36.16（间距 41.3），1:1 是 0 / 31（31.0），9:16 是 3.91 / 27.09（23.2）。绑定的出发点（推测，H3 没有公开消融）是让音频和视频 patch 共用同一条 w 尺，左耳在 RoPE 上离左半屏近，给声像和画面位置的对应提供先验。代价是 T2A 没有画布只能硬选一组值，S5 多画幅时立体声位置信号会偏移，而且文本固定在 w=0，离左耳近、离右耳远。S5 前要在三条里选：保持 H3 原样、T2A 按 S5 画幅集合随机抽 w、或者把音频 w 和画幅解绑。

用 DiffSynth 的 `MiniMaxH3DiT` 按预设在 meta device 上实例化数出的参数（AdaLN 含最终层）：

| 预设 | 层数 | hidden | 头 | FFN | time_emb | 总参数 | 主干 | AdaLN |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 2 | 256 | — | — | — | 只用来跑通管线 | | |
| `1b` | 24 | 1536 | 12×128 | 4608 | 768 | 1.320B | 0.736B | 0.513B（39%） |
| `10b` | 28 | 4096 | 32×128 | 12288 | 2048 | 10.822B | 6.107B | 4.247B（39%） |
| H3（参考） | 50 | 5376 | 56×128 | 14336 | 2688 | 33.123B | 19.268B | 13.039B（39%） |

AdaLN 每层是 `Linear(time_emb → hidden × 18)`，18 = 3 档模态 × 6 个调制量（两组 scale / shift / gate），所以参数随 time_emb 和 hidden 同时增长。10B 每个 token 实际参与计算的只有主干加 refiner 约 6.5B；AdaLN 前向几乎不花算力，但 Adam 状态要存约 68 GB。H3 的模态特异参数主要放在 AdaLN，不能直接砍，S2 在 1B 上对比独立 AdaLN 和共享式（adaLN-single 一类）。`text_dim` 从 5120 换到 2560 只差几 M。

## Flow matching 与采样（已实现，`chrome/flow/`）

- Rectified flow：`x_t = (1−σ)·x0 + σ·ε`，目标 `v = ε − x0`。
- 采样 σ：`σ = s·u / (1 + (s−1)·u)`，u 是均匀分布。音频 `s = 3`，H3 视频是 12（见 `audio_scheduler/` 和 `scheduler/`）。训练和推理用同一个 shift。
- AdaLN 的时间输入是 `t = 1 − σ`。参考行和干净行固定为 t = 1.0，不进损失。
- 采样器：Euler，η = 0。采样器按名字注册（`flow/sampler.py` 的 `register_sampler`），S4 的少步采样器加在旁边，配置 `preview.sampler` / `infer.sampler` 切换。
- CFG：训练时 10% 丢掉 caption，无条件输入是单个空格 `" "` 重新编码（1 个 token，H3 默认 negative_prompt 也是它），不是把 embedding 置零，也不用 `""`（0 个 token）。H3 自己的权重已经做过 CFG 蒸馏，我们从零训，S4 再蒸。

## 文本（已实现，`chrome/text/`）

H3 用 Qwen3-VL-32B（原 64 层，hidden 5120，vocab 151936）。本地 `${paths.h3_text_encoder}` 存的是完整权重（64 层语言模型 + 视觉塔 + lm_head，66.7 GB）；DiffSynth 加载时建 `Qwen3VLModel`（不带 lm_head），只建前 50 层（`num_retained_layers=50`），`language_model.norm` 换成 Identity，取第 50 层未过 norm 的 hidden states，维度 5120。50 层约 24.4B 参数，每 token 前向约 49 GFLOP。

用 H3 仓库自带的 tokenizer（`${paths.h3_tokenizer}`）。`tokenizer_config.json` 新增了 7 个 token（151669–151675）：`<d>`、`</d>`、`<|cutoff|>`、`<|lyrics_start|>`、`<|lyrics_end|>`、`<|caption_start|>`、`<|caption_end|>`；T2VA 的 caption 只用到 `<d>` / `</d>`。`[MUSIC]`、`[SPEECH]`、`<S>`、`<E>` 都是普通 BPE 切分。这 7 个 id 的 embedding 范数约 0.50，和没用到的 padding 行一样（正常 token 中位数约 1.15），说明 H3 没训练过它们，只当标记用。新增 token 只写在 `tokenizer_config.json` 里，单独读 `tokenizer.json` 会把 `<d>` 拆成字符，所以要经 `AutoTokenizer.from_pretrained(目录)` 加载（`chrome/data/captions.py`）。换更小的 Qwen3-VL 时把 H3 的 tokenizer 目录拷过去即可。

纯文本输入时视觉塔不参与，MRoPE 三轴位置相同、退化成 1D，T2A 阶段不加载视觉塔。

编码器按名字注册（`qwen3_vl`，单测用 `random`），外面套一层 `CachedTextEncoder`：内存 LRU → 磁盘（每条 caption 一个 safetensors，按编码器指纹分目录，原子写，多 rank 共享）→ 编码器。编码器在第一次缓存未命中时才加载，所以缓存填满的训练不会加载 Qwen。`scripts/text/cache_text.sh` 预填缓存，同时把前 `text_cache.stats_captions` 条的逐维 mean / std 写到 `<cache dir>/stats.safetensors`。

**第 50 层的异常值（实测）。** 这是 LLM 常见的 massive activation，不是个别 caption 的问题：每条 caption 的第 0 个 token 在第 731 维约 15872，整行 RMS 约 224；其它 token 的行 RMS 约 4.25，26600 行里只有 1 行超过 500。影响：第 731 列约占 `condition_proj` 自身梯度的 75%，但只占全模型梯度的 0.7%；refiner 最后的 RMSNorm 逐行归一化，进主干时这一行已经不突出。所以默认 `dit.text_norm: none`（和 H3 一样），另外有两个消融选项：

- `rmsnorm`：`condition_proj` 前逐行 RMSNorm。
- `stats`：用 `stats.safetensors` 逐维标准化，统计包含第 0 个 token（第 731 维的 std 因此变大，标准化后不会炸），可配 `dit.text_clip`（例如 8）截断。统计量存成 DiT 的 buffer，跟 checkpoint 走。

Qwen 规模主要是算力问题：截断 32B 编码一条约 200 token 的 caption 约 10 TFLOP，1B DiT 训一个 15 秒样本约 7 TFLOP，10B 约 55 TFLOP。离线缓存每条约 2 MB，1000 万条约 20 TB。1B 消融时比较更小的 Qwen3-VL 和截断后的 32B；保持 VL 系列，S5 才有现成的视觉语义通路。

## 数据管线（已实现）

清单 → 每个数据源一份 Arrow 索引 → 按权重混合、按 token 预算打包的 batch → 在线读波形。VAE 在训练时在线编码，不存 latent。全部参数在 `configs/data/t2a_base.yaml`，数据源列表在 `configs/data/sources_vrfoley.yaml`。

**索引**（`chrome/data/index.py`，`scripts/data/build_index.sh`）

- 清单是 JSONL 或 Parquet，可以给文件或目录。列名先用 `columns` 里配的，再退回 VR-Foley 的候选：音频 `path / audio / audio_filepath / wav`，caption `resp / caption / normalized_text / text`，时长 `duration / audio_duration`。
- JSONL 按字节切块（`data.build.chunk_mb`，默认 64 MB），Parquet 按 row group，多进程扫描。每行依次检查：keep 条件、音频路径、caption、preview 排除列表、坏文件列表、时长（清单里没有就读文件头）、长度策略、caption 不超过 512 token。每种丢弃原因都计数，写进 `meta.json`。
- 产物是 `outputs/data/index/<source>/index.arrow`（内存映射，DataLoader 的 worker 共享页缓存）和 `meta.json`。配置和清单文件（大小、mtime）都没变就跳过。改 `weight` / `task` 不用重建；改了影响行内容的字段（`max_records`、列名、keep、fit、long、时长范围、tokenizer、排除列表），loader 会报错，提示重建哪个源。
- 去重和验证集都看音频路径的 blake2b 哈希：同一路径只留第一条；哈希落在每百万 `holdout_ppm`（默认 500）以内的行留作验证，加数据不会改变已有行的归属。`max_records` 按哈希序截断，重建结果不变。

**长度策略**：只出整秒长度（4–15 s），一条样本的音频行数就是 80 × 秒。

- 超出整秒不到 `slack_seconds`（0.1 s）的尾巴直接裁掉。
- `fit: pad`（默认）：向上取整，补不到 1 秒的静音，语音和音效保留结尾。`fit: crop`（音乐）：向下取整，随机取窗。
- 超过 15.1 s：`long: drop`（默认，整段的 caption 对不上片段）或 `long: crop`（随机取 15 s 窗口）。

**采样**（`chrome/data/sampler.py`）

- 每个源用 `max_records` 按路径哈希序截成固定条数（重建结果不变）。不写 `weight` 时，抽源概率正比于留下的训练行数，所以混合比就是你写的条数。仍可写 `weight` 做额外加权。shard 是本 rank 在一个按 (seed, 源名, 源 epoch) 打乱的排列上取 `[rank::world]`，同一 epoch 内各 rank 不重叠。
- 打包：每条的代价是 80 × 秒 + caption token 数，累加到 `max_tokens`（默认 16384）就封包，放不下的那条留到下一个包，不丢样本。一个包里时长是混合的。
- 续训：sampler 给最近发出的每个 batch 存一份状态。训练器保存 `sampler.state_dict(batch["batch_id"])`，恢复时传给 `build_train_loader(..., state=...)`，从下一个 batch 接着出，取窗位置和 caption dropout 也一样（由每条样本的种子决定）。某个源的索引重建过，就只有这个源从下一个 epoch 重新开始。

**读取**（`chrome/data/dataset.py`、`batch.py`、`loader.py`）

- 要裁的文件只解码需要的那一段（soundfile seek，不行再用 torchaudio 的 `frame_offset`）；重采样核按采样率对缓存。之后转立体声 → 32 kHz → 裁或补到整秒 → 限峰值。
- 解码失败的文件追加进 `outputs/data/bad_audio.txt`（加文件锁；VR-Foley 的坏文件列表也一起读），换同源随机一条训练行。
- caption dropout 10%，无条件 caption 是 `" "`（token 220）。
- batch 里的 `waves` 是长度各异的列表。`encode_waves(vae, waves)` 只把等长的片段拼成一批编码：实测把短片段补零到长片段再编码，最后约 10 帧 latent 会变（最大差 1.08），所以不补零。
- 验证集：每个源取哈希最小的 `val.per_source`（256）条验证行，取中间窗口，不做 dropout，不跨源打包，每次是同一批。

训练器里这样接：

```python
from chrome.data.batch import encode_waves
from chrome.data.loader import build_train_loader

loader, sampler = build_train_loader(cfg, rank, world_size, state=ckpt.get("data"))
for batch in loader:
    latents = encode_waves(vae, batch["waves"], device)   # list of [2, 32, 40·s]
    ids = batch["input_ids"]                              # list of caption token ids
    ...
    ckpt["data"] = sampler.state_dict(batch["batch_id"])
```

加一个数据源：在 `data.sources` 里加一项（`name`、`paths`、`max_records`，按需写 `columns` / `keep` / `fit` / `long`），然后 `bash scripts/data/build_index.sh data.only=[新源名]`。`keep` 的值写成列表表示「取其中任一个」，例如 `qa_risk_level: [clean, medium]`。

**音频内嵌在 Parquet 里的数据（HF `datasets` 格式，如 WorldSpeech）**（`chrome/data/blobs.py`、`scripts/data/repack_audio.sh`）

- HF 分片把整个文件的音频放在一个 ZSTD 字典页里（每个文件 1 个 row group、1024 行、约 110 MB），读任意一条都要解压整列，没法按条随机读。所以先转存一次：音频字节原样拷进未压缩的 Arrow 文件 `<blobs>/<locale>/<分片名>.arrow`，不重编码。转存文件记录源分片的大小和 mtime，没变就跳过；建索引时发现过期或缺失会报错，提示重跑转存。
- 索引里的音频路径是引用 `<文件>.arrow#<行号>`；读取时内存映射取出这一条的字节，再走和普通文件相同的解码路径（soundfile，失败再用 torchaudio；要裁窗时在内存文件上 seek）。坏文件列表、去重、验证集划分照常按这个字符串处理。
- 建索引时不读音频列，只投影其余列。伴生 parquet（`sidecar`，`<root>/<locale>/<分片名>`）按行顺序 join：行数不同、`row_index` 或 `segment_id` 对不上都直接报错。同名列以伴生表为准。伴生文件的大小和 mtime 进清单指纹，改了就重建。
- 数据源新增三个键：`include`（目录里分片名的 glob，`train-*.parquet` 把 test 分片排除在外）、`sidecar`、`blobs`。没写这三个键的源，指纹和以前一样。
- 只含 WorldSpeech 的数据配置是 `configs/data/t2a_worldspeech.yaml`（源列表在 `sources_worldspeech.yaml`），用来先跑通新语种，再并进训练配比。

```bash
bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml
bash scripts/data/build_index.sh --config configs/data/t2a_worldspeech.yaml
bash scripts/data/inspect_data.sh --config configs/data/t2a_worldspeech.yaml
bash scripts/data/bench_loader.sh --config configs/data/t2a_worldspeech.yaml bench.output_dir=outputs/data/bench_ws
```

yue_hk 实测（2026-10-20，`max_seconds=15`）：169 个 train 分片、172,104 行，转存 17.4 GiB 用时 19 s（约 2 GB/s，32 进程）。建索引 91 s，留下 82,345 条训练行加 48 条验证行，共 297 h。丢弃：超过 15 s 的 72,935 条，`qa_risk_level=high` 的 16,560 条，短于 4 s 的 216 条。片段几乎都在 11–15 s，90% 需要补齐。loader 单进程 12 个 worker 跑到 7.8 batch/s、174 条/s，包填充率 97%。Opus 解码比读 wav 慢（同配置下 VR-Foley 五个源是 32 batch/s），但 10B 每步 3.4 s，仍有二十多倍余量。`resp` 里的 `[Cantonese][Cantonese]` 按原样保留。

### VR-Foley 五个源的实测（2026-10-12）

| 源 | 任务 | 清单行数 | 训练行 | 验证行 | 小时 | caption token p50 / p95 | 权重 | 按时长占比 |
|---|---|---|---|---|---|---|---|---|
| music | 音乐 | 4,322,339 | 4,300,539 | 2,201 | 11,936 | 151 / 167 | 0.9 | 9.9% |
| freesound | 音效 | 424,648 | 424,422 | 213 | 1,177 | 90 / 160 | 0.35 | 3.9% |
| audioset | 音效 | 490,000 | 489,540 | 279 | 1,356 | 141 / 222 | 0.35 | 3.9% |
| hifitts2 | 英文语音 | 11,311,053 | 8,648,398 | 4,349 | 23,649 | 41 / 61 | 3.5 | 38.1% |
| zhizhou | 中文语音 | 20,828,636 | 7,006,825 | 3,582 | 15,577 | 39 / 59 | 5.0 | 44.2% |

- 配比改成每个源自己的 `max_records`（music 90 万 / freesound 35 万 / audioset 35 万 / hifitts2 350 万 / zhizhou 500 万）。改数字后必须重建该源索引。zhizhou 全是 16 kHz（8 kHz 以上没有能量），按时长仍会偏多，S2 定配比时要重点看。
- 主要丢弃：zhizhou 的 keep 条件（`bw_flag=ok`、`cer≤0.3`）去掉 866 万条，短于 4 s 的 377 万条；HiFiTTS-2 超过 15 s 的 172 万条、短于 4 s 的 93 万条；音乐 caption 超过 512 token 的约 2 万条。
- 语音 88% 的片段是补齐的（平均补约 0.5 s 静音）；音乐、FreeSound、AudioSet 99% 是 10 s。
- 索引共 9.3 GB。五个源 64 进程 6 分钟建完；改成 64 MB 分块、`encode_batch_fast` 之后，FreeSound 从 77 s 降到 21 s。瓶颈是 caption 分词，不是 JSON 解析。
- loader 单进程 12 个 worker：32 batch/s、665 条/s，包的填充率 96%，各源实际占比和权重一致，约 4000 条里没有解码失败。读数据不会是瓶颈；在线 VAE 编码才是（15 s 立体声 fp32 约 148 ms，和 10B DiT 每条样本一步的计算量相当）。S2 实测：10B 每步 3.4 s 里 VAE 占 0.93 s（见「S2 训练框架」的实测）。

### 其它

- `configs/eval/preview_set.json` 共 18 条，从 VR-Foley 复制而来：音乐 3 条，英文语音 2 条，中文语音 9 条（普通话男女各年龄段、四川话、粤语、台湾腔、耳语、唱歌），FreeSound 2 条，AudioSet 2 条（鸟叫、雨声）。建索引时这些路径会被排除。对齐检查只用其中 8 条，见 `configs/vae/h3_frozen.yaml` 的 `clip_labels`。
- 16 kHz 的语音库升到 32 kHz 后，8 kHz 以上没有能量，配比过大会拉低高频。

## S2 训练框架（已实现）

分四层，每层只经一个接口和别的层打交道；扩展时只动一层：

| 层 | 位置 | 管什么 | 扩展方式 |
|---|---|---|---|
| Task | `chrome/tasks/` | 数据、条件怎么来（VAE / 文本 / packing）、loss、preview 渲染什么 | `@register_task("名字")`，配置 `task.kind`。S4 蒸馏、S5 视频各加一个 |
| Layout | `chrome/dit/layout.py` | 一条样本往 packed 序列里放哪些段（`[text \| audio L \| audio R]`） | `@register_layout`，配置 `layout.kind`。S5 加视频段，packer 和 DiT 不动 |
| Parallel | `chrome/train/parallel.py` | 模型怎么物化、初始化、包装、梯度怎么同步 | `ddp` / `fsdp`（`shard_size` 设了就是 HSDP），配置 `train.parallel` |
| Trainer | `chrome/train/trainer.py` | 优化器、LR、EMA、梯度累积、日志、验证、preview、checkpoint、续训 | 与任务无关，一般不用改 |

另外 `chrome/flow/sampler.py`（采样器注册）、`chrome/text/`（编码器注册）、`chrome/pipeline/t2a.py`（推理）各自独立。

`Task` 要实现的方法：`build_model`（在 meta device 上调用，冻结的部分在这里冻结）、`init_model`、`train_loader(rank, world, state)`、`val_loader`、`prepare(batch)`（返回模型输入、样本数、填充率、各阶段耗时）、`loss(model, inputs, generators)`（返回总 loss、每样本 loss、按组 loss）。可选：`grad_groups`（日志里分组看梯度范数）、`preview_items`、`preview`。latent 缓存以后只改 `T2ATask.prepare`。

**模型构建流程**：在 meta device 上建模型（10B 建好约 1 s，不占内存）→ `parallel.setup` 先分片（FSDP），再 `to_empty` 物化，再用同一个种子调用 `init_weights`。`init_weights` 覆盖所有参数和 buffer（RoPE 频率、text stats 也在里面），并且能处理 DTensor，所以 meta 上初始化和直接在设备上初始化结果一样（有单测）。

**FSDP2 的设定**（`fully_shard`，`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, cast_forward_inputs=False)`）：

- 每个 refiner block 和主干 block 各是一个分片单元，root 最后分片。fp32 主权重、梯度、Adam 状态都按分片存，计算用 bf16。
- `cast_forward_inputs=False`：RoPE 位置是 fp64、时间步要全精度，不能被自动转成 bf16。
- root 也设 `reshard_after_forward=True`：否则 no-grad 前向（验证、preview）之后，root 会一直留着已经 gather 的旧权重，EMA 换权重和下一次优化器更新都看不到。验证、preview、EMA 换权重前后还会显式 `parallel.reshard()`。
- 所有 rank 的前向次数必须一样（每次前向都有 all-gather）。验证集在各 rank 上分得不均时，batch 少的 rank 补跑重复的前向；preview 平均分到各 rank，不够的用最后一条补齐。
- EMA 只存本 rank 的分片（fp32），存盘时包成 DTensor。preview 用 `ema.swapped()` 临时换上 EMA 权重。
- HSDP：`shard_size=S` 时建 `("replicate","shard")` 二维 mesh，日志打 `hsdp R×S`。多机时设成 8，分片只在机内做，机间做数据并行。
- 梯度裁剪自己实现：各 rank 算本地平方和，在分片组上 all-reduce，再统一缩放。

DDP 设 `broadcast_buffers=False`，验证和 preview 走未包装的模型（DDP 每次前向都会同步 buffer，各 rank 次数不同时会卡住）。

**Checkpoint 格式**（`chrome/train/checkpoint.py`）：

```text
<run_dir>/config.yaml  log.jsonl  samples/step_*/
<run_dir>/checkpoints/latest                       最新一个完整的 step 目录名
<run_dir>/checkpoints/step_0001000/
    state/                torch.distributed.checkpoint（DCP）：model、ema（完整 model state，含 buffer）、optimizer
    trainer.pt            scheduler、step、dit 配置、并行方式、world size（rank 0 最后写，有它才算完整）
    data_rank*.pt         每个 rank 的 sampler 状态
    ema.safetensors       可选：train.export=ema 或 export_t2a.sh 导出的单文件 fp32 权重
```

DCP 加载时会自动重新分片，所以换卡数、DDP 和 FSDP 互换都能续训。单进程推理直接读 `state/`（不建进程组时 DCP 走 no_dist 模式），不必先导出。`load_weights` 优先读 `<weights>.safetensors`，没有就读 `state/[weights]`。`train.init_from` 只拿权重开新 run（不带优化器和数据状态），可以是 run 目录、step 目录或 .safetensors 文件。

**日志**（每 `log_every` 步，各 rank 平均）：loss（总体和每个源）、LR、梯度范数（总的和 `condition_proj` / `adaln` 分组）、每步秒数，拆成 `data`（等 loader）、`cond`（再分 `vae` / `text` / `pack`）、`model`（前向 + 反向）、每 rank 条数、填充率、显存峰值。`log.jsonl` 每条记录带 `batch_id`，可以和续训前后逐条对照。

### 实测（2026-10-13，单机 8 卡，每卡 140 GB）

条件阶段的耗时几乎全是 audio VAE：一个 16k 行的包（约 188 s 音频，按长度分 8 组编码）编码约 0.93 s；文本从缓存读约 10 ms，packing 约 3 ms。

| 跑法 | 结果 |
|---|---|
| tiny，500 条，DDP 8 卡，600 步（`t2a_smoke`） | train loss 1.26 → 0.33，val 0.39；preview 在 `outputs/train/t2a_smoke/samples/` |
| 端到端检查（`check_t2a_smoke.sh`，约 200 s） | 同一份初始权重下 DDP 和 FSDP 每步 loss 的相对差最大 4.4e-4，val 相对差 7e-5；FSDP 在第 10 步中断后续训，batch 完全相同，loss 相对差 2.1e-5；HSDP 2×2 和 `text_norm=stats` 都能训；从 `state/` 推理和从导出权重推理逐位相同 |
| 1.32B，FSDP 8 卡，16k token/卡 | 显存峰值 9.3 GB；稳定 1.57 s/步，其中 model 0.62 s、VAE 0.90 s（VAE 占 57%）；20 步 loss 1.23 → 0.53，val 0.53；checkpoint 写 7 s |
| 10.82B，FSDP 8 卡，16k token/卡，激活重算 | 显存峰值 34.8 GB；稳定后 3.4 s/步，其中 model 2.4 s、VAE 0.93 s；每卡约 20 条，填充率 0.97；第一步 31 s（预热）；10 步 loss 1.26 → 0.81；checkpoint 162 GB，写 41 s |
| 10.82B，关掉激活重算 | 显存不够（OOM）。保持 `gradient_checkpointing: true` |

10B 的 model 部分约 6.0e14 FLOP / 2.4 s，即每卡约 250 TFLOPS（不算重算）。在线 VAE 串行占了 10B 步时的 27%、1B 的 57%；做 latent 缓存或把 VAE 编码挪到另一个 stream 和 DiT 重叠，10B 约快 1.4 倍，1B 约快 2.5 倍。1B 消融前最值得先做这一项。

## 目录

已有：

```text
Chrome-10B/
├── skills.md
├── requirements.txt
├── docs/chrome_pipeline.html         本工程链路说明（数据处理 / VAE / DiT / RoPE / 条件）
├── docs/minimax_h3_analysis.html     H3 分析笔记
├── chrome/
│   ├── spec.py                       AudioSpec：秒 ↔ 采样点 ↔ 帧 ↔ 行
│   ├── config.py                     load_config：_base_ 继承 + 插值 + 命令行覆盖
│   ├── codec/
│   │   ├── audio_vae.py              AudioVAEConfig、ChromeAudioVAE
│   │   ├── whitening.py              LatentStats
│   │   ├── h3_io.py                  读 bundle / diffusers，导出为 diffusers
│   │   └── modules/                  conv、activations、alias_free、encoder、attn_proj、gaussian、bigvgan
│   ├── data/
│   │   ├── audio_io.py               read_audio、read_audio_window、resample（核缓存）、to_stereo、limit_peak、load_audio、save_audio
│   │   ├── manifest.py               JSONL / Parquet 清单：列名候选、keep 过滤、分块、伴生表 join
│   │   ├── blobs.py                  Parquet 内嵌音频 → Arrow 转存，`<文件>.arrow#<行>` 引用的读取
│   │   ├── captions.py               H3 tokenizer 加载、caption → ids、NULL_CAPTION
│   │   ├── index.py                  LengthPolicy、SourceSpec、并行建索引、SourceIndex
│   │   ├── sampler.py                MixtureBatchSampler（加权混合、token 打包、续训）、验证集用的固定 batch
│   │   ├── dataset.py                T2ADataset、BadAudioList
│   │   ├── batch.py                  collate_t2a、encode_waves（等长分组编码）
│   │   ├── loader.py                 build_train_loader、build_val_loader
│   │   └── cli.py                    build / inspect / bench
│   ├── text/
│   │   ├── base.py                   TextEncoder 接口、注册表、build_text_encoder
│   │   ├── qwen.py                   截断的 Qwen3-VL（前 N 层，末层不过 norm）
│   │   ├── random.py                 固定随机表（单测、空跑）
│   │   └── cache.py                  CachedTextEncoder：内存 LRU → 磁盘 → 编码器（懒加载）
│   ├── dit/
│   │   ├── config.py                 DiTConfig（预设的 schema；text_norm / text_clip）
│   │   ├── packing.py                Segment、pack：多样本拼一条 varlen 序列，补到 64 的倍数
│   │   ├── layout.py                 Layout 注册表；T2ALayout = [text | audio L | audio R]
│   │   ├── rope.py                   MMRoPE、StereoW（两耳的 w）
│   │   ├── adaln.py                  PerLayerAdaLN / SharedAdaLN、AdaLNTable
│   │   ├── attention.py              varlen 注意力（flash / sdpa）
│   │   ├── layers.py                 DiT block、token refiner
│   │   └── model.py                  ChromeDiT（init_weights、shard_units、set_text_stats）
│   ├── flow/
│   │   ├── schedule.py               shift、TimestepSampler、inference_sigmas
│   │   ├── objective.py              x_σ、v = ε − x0
│   │   └── sampler.py                Sampler 注册表、EulerSampler
│   ├── tasks/
│   │   ├── base.py                   Task 接口、注册表、Generators / LossOutput / Prepared
│   │   └── t2a.py                    T2ATask、encode_t2a、t2a_flow_loss
│   ├── pipeline/t2a.py               T2APipeline：from_run、generate（CFG、每条独立种子）
│   ├── train/
│   │   ├── dist.py                   torchrun 环境、单进程退化、all_reduce_max
│   │   ├── parallel.py               DDPParallel、FSDPParallel（含 HSDP）、分片 EMA
│   │   ├── checkpoint.py             run 目录、DCP 存取、load_weights、export_weights
│   │   ├── trainer.py                与任务无关的 Trainer
│   │   └── cli.py                    train / infer / text-cache / export
│   └── eval/recon.py                 si_sdr、mel_l1、log_stft_l1、highband_ratio、stereo_report、clip_metrics
├── configs/
│   ├── _base_/{audio_spec,paths}.yaml
│   ├── vae/model_h3.yaml             H3 VAE 结构，键名对应 AudioVAEConfig 的字段
│   ├── vae/h3_frozen.yaml            冻结的 H3 VAE + 对齐检查参数
│   ├── data/t2a_base.yaml            数据管线参数（索引、长度、loader、验证集、bench）
│   ├── data/sources_vrfoley.yaml     VR-Foley 的五个数据源和权重
│   ├── data/sources_worldspeech.yaml WorldSpeech 各语种（内嵌音频 + QA 伴生表），t2a_worldspeech.yaml 单独使用
│   ├── dit/{tiny,1b,10b}.yaml        DiT 预设
│   ├── text/{qwen3vl_32b,random}.yaml
│   ├── train/t2a_base.yaml           task / layout / flow / train / preview / infer / text_cache 全部参数
│   ├── train/t2a_smoke.yaml          tiny + 500 条，DDP
│   ├── train/t2a_1b.yaml             1B 消融基线，FSDP
│   ├── train/t2a_10b.yaml            10B，FSDP（多机时 shard_size=8 即 HSDP）
│   └── eval/preview_set.json
├── scripts/
│   ├── env.sh                        PYTHON、PYTHONPATH
│   ├── vae/check_h3_parity.sh
│   ├── data/{repack_audio,build_index,inspect_data,bench_loader,check_data_pipeline}.sh
│   ├── text/cache_text.sh            预填文本缓存 + 写 stats.safetensors
│   ├── dit/check_t2a_stack.sh        S2 单测（CPU 即可）
│   ├── train/train_t2a.sh            torchrun 启动，单机 / 多机
│   ├── train/export_t2a.sh           checkpoint → 单文件 safetensors
│   ├── train/check_t2a_smoke.sh      S2 端到端检查（8 卡）
│   └── infer/infer_t2a.sh
├── tests/{test_h3_parity,test_data_pipeline,test_t2a_stack,test_t2a_e2e}.py
└── outputs/                          运行产物；数据索引 outputs/data/index/，文本缓存 outputs/text_cache/，训练 outputs/train/

## 配置和脚本约定

- 配置用 OmegaConf。`_base_` 可以是单个路径或列表，按顺序合并，自身最后合并；然后应用命令行的 `key=value` 覆盖，再解析 `${...}` 插值。
- 配置里的相对路径从 `Chrome-10B/` 根目录算起。脚本先 `cd` 到根目录再 `source scripts/env.sh`，所以从哪里调用都可以。
- 每个脚本的头部都写可以直接复制运行的命令。额外参数原样透传成配置覆盖。
- 外部路径只写在 `configs/_base_/paths.yaml` 里：H3 权重根目录、两种 VAE 目录、文本编码器、tokenizer、DiffSynth、VR-Foley、音频数据根目录、preview 清单、输出目录、数据索引目录。
- 列表里有 `${...}` 插值时写成块列表（`- ${paths.x}`）。写成 `[${paths.x}]` 时 YAML 会把 `{` 当成映射，解析失败。

## 已知坑

- 系统自带的 `python3` 没有 safetensors，要用 ovi 环境（`scripts/env.sh` 里的 `PYTHON`）。
- 直接 `import diffsynth` 会经 `diffsynth/core/loader/config.py` 去导入 modelscope，而 ovi 里没装。对齐检查的做法是：在 `sys.modules` 里放空的 `diffsynth`、`diffsynth.core`、`diffsynth.models` 和一个 `diffsynth.core.attention`（einops 重排加 SDPA），然后只加载 `minimax_h3_audio_vae.py` 这一个文件。DiffSynth 加载权重前，`pre_block.attn.{q_bias,v_bias,zero_k_bias}` 要加 `.weight` 后缀。
- 官方 bundle 的 `MiniMaxH3AudioVAE` 只暴露 `decode`；要编码，得手动走 `.model` 的 `encoder → pre_block → mean_proj`。
- 做对齐时要关掉 TF32。打开时 decode 对比官方的误差是 1.1e-3，encode 对比 DiffSynth 是 4.9e-2；关掉后都在 1e-4 以内。
- GAN 解码器的重建质量要看频谱指标（mel_l1），不要看 SI-SDR。
- VAE 批量编码只能拼等长片段。补零后一起编码，被补的片段最后约 10 帧 latent 会变（编码器的感受野伸进了补零区）。
- H3 tokenizer 必须经 `AutoTokenizer.from_pretrained(目录)` 加载，直接读 `tokenizer.json` 会把 `<d>` 拆开，id 全错。
- 建索引的时间主要花在 caption 分词上；用 `encode_batch_fast`（不算 offset），速度约 2.7 倍，id 完全相同。
- FSDP 下所有 rank 的前向次数必须相同，否则卡在 all-gather 直到超时（`train.dist_timeout_minutes`）。任务里新加的验证、preview 逻辑都要按这个补齐。
- FSDP 的 `MixedPrecisionPolicy` 默认会把前向输入转成 bf16，RoPE 位置和时间步会因此丢精度，所以要设 `cast_forward_inputs=False`。
- root 模块不 reshard 的话，no-grad 前向之后会一直留着 gather 好的旧权重，EMA 换权重会失效。
- 10B 关掉激活重算会 OOM（16k token/卡）。
- 带 `log.jsonl` 的 run 目录重跑同一命令会续训（`train.resume=auto`）；要从头开始，加 `train.resume=null` 或换 `train.run_name`。

## 环境与命令

用 conda 环境 `ovi`（torch 2.6、torchaudio 2.6、omegaconf 2.3），依赖见 `requirements.txt`。

```bash
cd Chrome-10B

# S0：H3 Audio VAE 对齐检查（约 15 秒，单卡）
CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh

# 只跑一条 4 秒的音频，输出另放一处
CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh parity.clip_seconds=4 parity.clip_labels='[solo piano]' parity.output_dir=outputs/vae/parity_smoke

# 换解释器
PYTHON=/path/to/python bash scripts/vae/check_h3_parity.sh

# S1：数据管线自检（合成音频，CPU，约 25 秒）
bash scripts/data/check_data_pipeline.sh

# 建索引：没变的源自动跳过；只重建某个源
bash scripts/data/build_index.sh
bash scripts/data/build_index.sh data.only=[zhizhou] data.build.force=true

# 每个源的行数、时长、丢弃原因、长度分布、混合后的占比和 epoch 数
bash scripts/data/inspect_data.sh inspect.world_size=64 inspect.steps=300000

# loader 吞吐，顺便写出几条音频和 caption 到 outputs/data/bench/ 试听；加 VAE 编码要一张空闲的卡
bash scripts/data/bench_loader.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/data/bench_loader.sh bench.vae=true

# S2：单测（CPU，约 15 秒）和端到端检查（8 卡，约 200 秒；--keep 保留产物）
CUDA_VISIBLE_DEVICES="" bash scripts/dit/check_t2a_stack.sh
bash scripts/train/check_t2a_smoke.sh --keep

# 预填文本缓存并写 text stats（一张卡，第一次会加载 Qwen3-VL-32B/50）
CUDA_VISIBLE_DEVICES=0 bash scripts/text/cache_text.sh configs/train/t2a_smoke.yaml

# 训练：默认用所有可见卡；同一命令重跑即续训
bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml                                   # tiny，500 条，DDP
bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml train.parallel.kind=fsdp train.run_name=t2a_smoke_fsdp
bash scripts/train/train_t2a.sh configs/train/t2a_1b.yaml                                      # 1B，FSDP
bash scripts/train/train_t2a.sh configs/train/t2a_1b.yaml dit.adaln=shared train.run_name=t2a_1b_shared
bash scripts/train/train_t2a.sh configs/train/t2a_1b.yaml dit.text_norm=stats dit.text_clip=8 train.run_name=t2a_1b_stats
bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml data.loader.subset=100 train.max_steps=20 train.run_name=t2a_10b_dry
bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml                                     # 10B，单机 8 卡 FSDP
# 多机（每台机器各跑一次，NODE_RANK 不同）：机内分片、机间复制（HSDP）
NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml train.parallel.shard_size=8

# 推理（直接读分片 checkpoint，默认 latest 的 EMA）
CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_smoke 'infer.prompts=["Heavy rain on a tin roof."]'
CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_smoke infer.prompt_file=configs/eval/preview_set.json infer.steps=30

# 导出单文件 fp32 权重：<run>/checkpoints/<step>/ema.safetensors
bash scripts/train/export_t2a.sh outputs/train/t2a_10b
bash scripts/train/export_t2a.sh outputs/train/t2a_10b --checkpoint 2000 --weights model
```

在代码里使用：

```python
from chrome.codec import load_h3_audio_vae

vae = load_h3_audio_vae("/mnt/data04/480634/public/2_CKPT/3_PreTRAINED/MiniMax-H3/FL2VA/audio_vae", device="cuda")
latent = vae.encode_audio(wave)      # [2, N] -> [2, 32, T]，已白化
wave_rec = vae.decode_audio(latent)  # [2, 32, T] -> [2, T*800]
```
