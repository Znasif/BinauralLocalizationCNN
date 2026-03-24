import numpy as np


def fold_front_back(az_deg):
    """Map azimuth (0-360) to 0-90 by reflecting across median and coronal planes."""
    az = az_deg % 360
    az = np.where(az > 180, 360 - az, az)
    az = np.where(az > 90, 180 - az, az)
    return az


def calc_mae(preds, labels):
    """Return (err_az_sum, err_el_sum, err_az_fb_sum) for a batch.

    Classes: 504 total = 7 elevations × 72 azimuths
      az_bin = class % 72  → degrees = az_bin * 5
      el_bin = class // 72 → degrees = el_bin * 10
    """
    p_az = (preds  % 72) * 5
    p_el = (preds  // 72) * 10
    l_az = (labels % 72) * 5
    l_el = (labels // 72) * 10

    diff_az = np.abs(p_az - l_az)
    err_az   = np.minimum(diff_az, 360 - diff_az)
    err_az_fb = np.abs(fold_front_back(p_az) - fold_front_back(l_az))
    err_el   = np.abs(p_el - l_el)

    return err_az.sum(), err_el.sum(), err_az_fb.sum()


def split_accuracy(preds, labels, click_types):
    """Returns (acc_0click, acc_1click, n_0click, n_1click)."""
    correct = (preds == labels)
    m0 = (click_types == 0)
    m1 = (click_types == 1)
    acc0 = float(correct[m0].mean()) if m0.any() else float('nan')
    acc1 = float(correct[m1].mean()) if m1.any() else float('nan')
    return acc0, acc1, int(m0.sum()), int(m1.sum())


def print_epoch_table(label, epoch_str, rows):
    """Print a bordered table for epoch-end metrics.

    rows: list of dicts with keys:
        name     : str   ('all', '0-clk', '1-clk')
        loss     : float or None
        acc      : float (0-1)
        n_ok     : int
        n        : int
        mae_az   : float (degrees)
        mae_azfb : float (degrees)
        mae_el   : float (degrees)
    float('nan') values are displayed as '---'.
    """
    def f_loss(v):
        return f'{v:6.2f}' if v is not None and v == v else '  ----'
    def f_acc(v):
        return f'{v*100:6.2f}' if v == v else '   ---'
    def f_cnt(ok, n):
        return f'{f"{ok}/{n}":>10s}'
    def f_deg(v):
        return f'{v:6.1f}' if v == v else '   ---'

    col = (f'  │  {"":8s} {"Loss":>6s}  {"Acc%":>6s}  {"n_ok/n":>10s}'
           f'  {"Az":>6s}  {"AzFB":>6s}  {"El":>6s}  │')
    W = len(col) - 4
    hdr = f' {label} {epoch_str} '
    top = f'  ┌─{hdr}{"─" * (W - len(hdr) - 1)}┐'
    bot = f'  └{"─" * W}┘'

    print(top)
    print(col)
    for r in rows:
        line = (f'  │  {r["name"]:8s}'
                f' {f_loss(r.get("loss"))}'
                f'  {f_acc(r["acc"])}'
                f'  {f_cnt(r["n_ok"], r["n"])}'
                f'  {f_deg(r["mae_az"])}'
                f'  {f_deg(r["mae_azfb"])}'
                f'  {f_deg(r["mae_el"])}'
                f'  │')
        print(line)
    print(bot)
