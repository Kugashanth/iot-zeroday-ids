import numpy as np
import pytest

from iotids.data import BENIGN, Preprocessor, family_of, make_splits


@pytest.mark.parametrize("label,fam", [
    ("BenignTraffic", BENIGN), ("DDoS-ICMP_Flood", "DDoS"), ("DoS-UDP_Flood", "DoS"),
    ("Mirai-greeth_flood", "Mirai"), ("Recon-PortScan", "Recon"), ("VulnerabilityScan", "Recon"),
    ("MITM-ArpSpoofing", "Spoofing"), ("DNS_Spoofing", "Spoofing"), ("SqlInjection", "Web"),
    ("XSS", "Web"), ("DictionaryBruteForce", "BruteForce"), ("Backdoor_Malware", "Web"),
])
def test_family_mapping(label, fam):
    assert family_of(label) == fam


def test_ddos_not_confused_with_dos():
    assert family_of("DDoS-SYN_Flood") == "DDoS"
    assert family_of("DoS-SYN_Flood") == "DoS"


def test_splits_are_zero_day(small_df):
    cfg = {"seed": 0, "split": {"train_frac": 0.6, "val_frac": 0.2, "unknown_families": ["Mirai", "Web"]}}
    sp = make_splits(small_df, cfg)
    assert set(sp.train["family"]) == {BENIGN}
    assert set(sp.val["family"]) == {BENIGN}
    assert set(sp.test["family"]) == {BENIGN, "Mirai", "Web"}
    assert "Mirai" not in set(sp.known["family"])
    # no benign row leaks between train / val / test
    idx = lambda d: set(map(tuple, d.drop(columns=["family", "label"]).round(6).values))  # noqa: E731
    assert not idx(sp.train) & idx(sp.test[sp.test["family"] == BENIGN])


def test_preprocessor_fit_on_benign_only_and_handles_inf(small_df):
    benign = small_df[small_df["family"] == BENIGN]
    pre = Preprocessor(drop_cols=["ts"]).fit(benign)
    X = pre.transform(small_df)
    assert "ts" not in pre.feature_names_
    assert np.isfinite(X).all()
    assert X.dtype == np.float32
    Xb = pre.transform(benign)
    assert np.allclose(Xb.mean(0), 0, atol=0.2)
    assert np.abs(X).max() <= 10.0
