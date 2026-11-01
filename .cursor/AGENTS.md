# AGENTS.md — Olympic Time Series Contest Kit

این فایل خلاصهٔ تصمیم‌ها، خروجی کار، و قرارداد کار برای agentهای بعدی است. جزئیات مفهومی سری‌زمانی را از [`time_series.md`](time_series.md) بخوان؛ شکل سوالات مسابقهٔ قبلی در [`Questions/`](Questions/).

## هدف پروژه

آماده‌سازی کیت قابل‌حمل برای مسابقهٔ دیتا با محور **پیش‌بینی سری زمانی univariate**. مجازیم کد آماده ببریم. مسیر پیش‌فرض روز مسابقه: پیش‌پردازش → FE → CV → train → inference → submission.

## تصمیم‌های قفل‌شده (grilling)

| موضوع | تصمیم |
|---|---|
| شکل تحویل | پکیج `tskit` + نوت‌بوک نازک نبرد |
| خانواده مسئله | فقط **univariate**؛ covariate برون‌زا اختیاری |
| مدل v1 | فقط **XGBoost** + FE زمانی |
| خارج از v1 | ARIMA / Prophet / PatchTST / پنل چندسری / تشخیص خودکار نام ستون |
| افق پیش‌بینی | `recursive` (پیش‌فرض) و `direct` با سوییچ |
| تفاضل هدف | `target_diff=True` پیش‌فرض؛ قابل خاموش شدن |
| FE پیش‌فرض | lags `[1,7,14]`؛ rolling mean/std `[7,14]` + `shift(1)`؛ sin/cos تقویمی |
| ستون‌ها | `date_col` و `target_col` صریح در کانفیگ؛ بدون auto-detect نام |
| تقویم داده | `freq` اجباری → reindex؛ NaNِ `y` برای هم‌ترازی تقویمی نگه داشته می‌شود و فقط هنگام ساخت X/y حذف می‌شود؛ exog با ffill محدود |
| CV | walk-forward با `n_splits` / `gap` / `horizon`؛ متریک پیش‌فرض MAE (+ RMSE/RMSLE/MAPE)؛ خطای per-horizon |
| HPO | defaults قوی در `fit`؛ `tune()` اختیاری و کوتاه (Optuna یا RandomizedSearch) |
| دمو | دیتای مصنوعی در `data/synthetic/` + نوت‌بوک end-to-end |

پلن پیاده‌سازی: [`.cursor/plans/ts_contest_toolkit_3c8c197e.plan.md`](.cursor/plans/ts_contest_toolkit_3c8c197e.plan.md)

## نقشهٔ ریپو

```
tskit/                     # پکیج contest-ready
  config.py                # TSConfig
  preprocess.py            # sort / reindex / NaN policy
  stationarity.py          # ADF / KPSS (فقط تشخیصی)
  features.py              # calendar + lag/rolling leakage-safe
  model.py                 # XGBForecaster: fit / predict / tune
  cv.py                    # walk_forward_validate + summarize_cv
  submit.py                # to_submission
notebooks/battle_notebook.ipynb
data/synthetic/            # train.csv, test.csv, truth.csv (محلی)
outputs/                   # submission روز تمرین
requirements.txt
time_series.md             # خلاصه ویدئوی آموزشی مسابقه
Questions/                 # سوالات قبلی (غیر TS؛ الگوی نوت‌بوک/submission)
```

API عمومی: `TSConfig`, `prepare`, `prepare_train_test`, `add_features`, `check_stationarity`, `XGBForecaster`, `walk_forward_validate`, `summarize_cv`, `to_submission`.

## روز مسابقه — ترتیب کار

1. مسیر `TRAIN_PATH` / `TEST_PATH` و فیلدهای `TSConfig` را در [`notebooks/battle_notebook.ipynb`](notebooks/battle_notebook.ipynb) تنظیم کن (`date_col`, `target_col`, `freq`, `horizon`, `exog_cols`, `metric`).
2. `prepare_train_test` → `check_stationarity` (تشخیصی؛ مسیر fit را بلاک نکن).
3. `walk_forward_validate(..., strategies=["recursive","direct"])` → استراتژی بهتر را بردار.
4. اگر روند ضعیف بود، یک بار با `target_diff=False` هم CV بگیر.
5. اختیاری: `model.tune(train, cfg, n_trials=40)`.
6. `fit` → `predict` → `to_submission`؛ ستون‌های submission را با فرمت رسمی مسابقه هم‌تراز کن (`id` / `date` / فقط prediction).

## قواعد فنی که نباید بشکنند

- **Leakage:** lag و rolling فقط از گذشته؛ rolling با `shift(1)`.
- **Recursive:** بعد از هر گام، پیش‌بینی وارد تاریخچه می‌شود تا lag بعدی ساخته شود.
- **Direct:** یک مدل XGB جدا برای هر `h` در `1..H`.
- **target_diff:** آموزش روی `y_t - y_{t-1}` (یا برای direct: `y_{t+h} - y_t`)؛ در infer بازسازی سطح.
- **Reindex قبل از FE:** حذف زودهنگام ردیف‌های NaNِ `y` باعث می‌شود lag بر اساس ردیف باشد نه تقویم — ممنوع.
- ایستایی فقط diagnostic است؛ تصمیم تفاضل با فلگ `target_diff` است نه نتیجهٔ ADF/KPSS به‌صورت خودکار.

## الگوی سوالات قبلی (برای تطبیق submission)

سوالات داخل `Questions/` سری‌زمانی نیستند، ولی قالب مسابقه این است: روایت فارسی، معمولاً `train`/`test`، نوت‌بوک، خروجی `submission.csv` یا `result.zip`، متریک سفارشی. کیت را با همان الگوی «نوت‌بوک نازک + منطق در پکیج» ساختیم.

## وابستگی‌ها

```bash
pip install -r requirements.txt
```

اصلی‌ها: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `statsmodels`, `optuna` (اختیاری برای tune).

## وقتی agent باید این فایل را دنبال کند

- تغییر یا گسترش `tskit` / نوت‌بوک نبرد / FE / CV / inference
- آماده‌سازی برای روز مسابقهٔ time series
- هر درخواست «کد آماده ببریم» مرتبط با forecasting در این ریپو

اگر تصمیم تازه‌ای یکی از ردیف‌های جدول قفل‌شده را عوض کرد، همان ردیف را اینجا به‌روز کن؛ کد را با جدول هم‌تراز نگه دار.
