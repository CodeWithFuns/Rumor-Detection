from rumor_detection.preprocess import clean_text

def test_clean_urls_mentions_hashtags_emojis():
    s = "WOW 😅 visit https://x.com/abc @user #Breaking"
    out = clean_text(s)
    # url removed, emojis/punct stripped, no @ or #, lowercased
    assert "http" not in out and "@" not in out and "#" not in out
    assert "wow" in out and "visit" in out and "breaking" in out
