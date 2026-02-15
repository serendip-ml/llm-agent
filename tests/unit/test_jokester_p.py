"""Tests for jokester-p agent components."""

from unittest.mock import MagicMock

import pytest

from llm_agent.agents.jokester_p.novelty import NoveltyCheck, NoveltyChecker
from llm_agent.agents.jokester_p.pairing import PreferencePair, RatedJoke


pytestmark = pytest.mark.unit


class TestNoveltyCheck:
    """Tests for NoveltyCheck dataclass."""

    def test_novel_joke(self):
        """NoveltyCheck for a novel joke."""
        check = NoveltyCheck(is_novel=True, max_similarity=0.3, similar_joke=None)
        assert check.is_novel is True
        assert check.max_similarity == 0.3
        assert check.similar_joke is None

    def test_duplicate_joke(self):
        """NoveltyCheck for a duplicate joke."""
        check = NoveltyCheck(is_novel=False, max_similarity=0.95, similar_joke="existing joke")
        assert check.is_novel is False
        assert check.max_similarity == 0.95
        assert check.similar_joke == "existing joke"


class TestNoveltyChecker:
    """Tests for NoveltyChecker."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def mock_learn_trait(self):
        """Create mock learn trait."""
        trait = MagicMock()
        trait.has_embedder = True
        return trait

    def test_has_embedder_delegates_to_trait(self, mock_logger, mock_learn_trait):
        """has_embedder property delegates to learn trait."""
        checker = NoveltyChecker(mock_logger, mock_learn_trait, similarity_threshold=0.85)
        assert checker.has_embedder is True

        mock_learn_trait.has_embedder = False
        assert checker.has_embedder is False

    def test_check_novel_no_existing_jokes(self, mock_logger, mock_learn_trait):
        """Check returns novel when no existing jokes found."""
        mock_learn_trait.recall.return_value = []
        checker = NoveltyChecker(mock_logger, mock_learn_trait, similarity_threshold=0.85)

        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.0
        assert result.similar_joke is None

    def test_check_novel_below_threshold(self, mock_logger, mock_learn_trait):
        """Check returns novel when similarity is below threshold."""
        similar = MagicMock()
        similar.score = 0.5
        similar.entity.content = "Different joke"
        mock_learn_trait.recall.return_value = [similar]

        checker = NoveltyChecker(mock_logger, mock_learn_trait, similarity_threshold=0.85)
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.5
        assert result.similar_joke == "Different joke"

    def test_check_duplicate_above_threshold(self, mock_logger, mock_learn_trait):
        """Check returns not novel when similarity exceeds threshold."""
        similar = MagicMock()
        similar.score = 0.92
        similar.entity.content = "Very similar joke"
        mock_learn_trait.recall.return_value = [similar]

        checker = NoveltyChecker(mock_logger, mock_learn_trait, similarity_threshold=0.85)
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 0.92
        assert result.similar_joke == "Very similar joke"

    def test_check_fails_closed_on_error(self, mock_logger, mock_learn_trait):
        """Check returns not novel on error (fail closed)."""
        mock_learn_trait.recall.side_effect = Exception("Database error")

        checker = NoveltyChecker(mock_logger, mock_learn_trait, similarity_threshold=0.85)
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 1.0


class TestPreferencePair:
    """Tests for preference pairing types."""

    def test_rated_joke(self):
        """RatedJoke stores joke data."""
        joke = RatedJoke(id=1, content="Funny joke", stars=4)
        assert joke.id == 1
        assert joke.content == "Funny joke"
        assert joke.stars == 4

    def test_preference_pair_margin(self):
        """PreferencePair margin is star difference."""
        chosen = RatedJoke(id=1, content="Great joke", stars=5)
        rejected = RatedJoke(id=2, content="Bad joke", stars=1)
        pair = PreferencePair(chosen=chosen, rejected=rejected)

        assert pair.margin == 4

    def test_preference_pair_zero_margin(self):
        """PreferencePair with same stars has zero margin."""
        chosen = RatedJoke(id=1, content="Joke A", stars=3)
        rejected = RatedJoke(id=2, content="Joke B", stars=3)
        pair = PreferencePair(chosen=chosen, rejected=rejected)

        assert pair.margin == 0
