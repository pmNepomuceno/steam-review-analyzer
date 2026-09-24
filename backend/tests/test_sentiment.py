import joblib
import pytest

from scripts import train_sentiment
from scripts.train_sentiment import Review
from src.ml import sentiment
from src.ml.exceptions import SentimentModelNotFound

POSITIVE_TEXTS = [
    "amazing game, loved it", "great fun and amazing art", "best game ever, amazing",
    "loved every minute, great", "fantastic and fun", "great story, amazing music",
    "fun fun fun, loved it", "amazing, highly recommend", "great value, fantastic",
    "loved the gameplay, great",
]
NEGATIVE_TEXTS = [
    "terrible game, refund", "boring and broken", "awful bugs, waste of money",
    "broken mess, terrible", "boring, refund it", "crashes constantly, awful",
    "waste of time, boring", "terrible performance, broken", "awful, do not buy",
    "refund, broken and boring",
]


def corpus(n_pos: int, n_neg: int) -> list[Review]:
    pos = [Review("Pos Game", POSITIVE_TEXTS[i % 10], True) for i in range(n_pos)]
    neg = [Review("Neg Game", NEGATIVE_TEXTS[i % 10], False) for i in range(n_neg)]
    return pos + neg


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    result = train_sentiment.train_and_evaluate(corpus(40, 40))
    path = tmp_path_factory.mktemp("models") / "sentiment.joblib"
    train_sentiment.save(result, path)
    return path


@pytest.fixture(autouse=True)
def loaded_model(model_path):
    sentiment.load_model(model_path)
    yield
    sentiment._artifact = None


def assert_valid(pred: dict) -> None:
    assert set(pred) == {"label", "confidence"}
    assert pred["label"] in {"positive", "negative"}
    assert type(pred["confidence"]) is float
    assert 0.0 <= pred["confidence"] <= 1.0


def test_predict_positive():
    pred = sentiment.predict("this game is amazing")
    assert_valid(pred)
    assert pred["label"] == "positive"
    assert pred["confidence"] > 0.5


def test_predict_negative():
    pred = sentiment.predict("broken and boring, refund")
    assert_valid(pred)
    assert pred["label"] == "negative"


@pytest.mark.parametrize(
    "text",
    [
        "10/10",
        "",
        "   ",
        "🔥🔥🔥 👍",
        "このゲームは最高",
        "Ça marche très bien",
        "g̷̢l̶i̵t̸c̷h̶ \x00\x07\u200b",
        "a" * 20_000,
    ],
)
def test_predict_never_crashes(text):
    assert_valid(sentiment.predict(text))


def test_confidence_is_raw_probability(model_path):
    # Vocabulary the model never saw: the probability sits near the intercept, and
    # predict() must report it as-is rather than rounding it away.
    text = "the quick brown fox"
    pred = sentiment.predict(text)
    expected = joblib.load(model_path)["pipeline"].predict_proba([text])[0].max()
    assert pred["confidence"] == float(expected)
    assert 0.5 <= pred["confidence"] < 1.0


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(SentimentModelNotFound):
        sentiment.load_model(tmp_path / "nope.joblib")


def test_drop_empty_excludes_whitespace_reviews():
    reviews = [Review("G", "good", True), Review("G", "   ", True), Review("G", "", False)]
    kept, excluded = train_sentiment.drop_empty(reviews)
    assert [r.text for r in kept] == ["good"]
    assert excluded == 2


def test_report_on_one_sided_game_warns_and_compares_to_baseline():
    reviews = corpus(970, 30)  # 97% positive
    result = train_sentiment.train_and_evaluate(reviews)
    report = train_sentiment.format_report(result, reviews, n_excluded=0)

    assert result.minority == "negative"
    assert result.baseline["classes"]["negative"]["f1"] == 0.0
    for cls in ("negative", "positive"):
        for metric in ("precision", "recall", "f1"):
            assert any(line.split()[:2] == [cls, metric] for line in report.splitlines())
    assert "WARNING: the test set is 97.0% one class" in report
    assert "only 6 negative reviews in the test set" in report
    assert "PASS:" in report or "FAIL:" in report


def test_report_says_fail_plainly_when_model_does_not_beat_baseline():
    reviews = corpus(95, 5)
    result = train_sentiment.train_and_evaluate(reviews)
    result.model["classes"]["negative"]["f1"] = 0.0
    result.beats_baseline = False
    report = train_sentiment.format_report(result, reviews, n_excluded=0)
    assert "FAIL: model does NOT beat the majority baseline on the minority class" in report


def test_training_needs_both_classes():
    with pytest.raises(ValueError, match="each class"):
        train_sentiment.train_and_evaluate(corpus(50, 1))


def test_write_report_saves_header_and_body(tmp_path):
    path = tmp_path / "eval" / "sentiment_report.txt"
    train_sentiment.write_report("PASS: something", path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("Sentiment model report, generated ")
    assert text.rstrip().endswith("PASS: something")
