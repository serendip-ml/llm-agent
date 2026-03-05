"""Tests for jokester-p agent components."""

from unittest.mock import MagicMock

import pytest
from appinfra import DotDict

from llm_gent.agents.jokester_p.history import JokeHistory
from llm_gent.agents.jokester_p.novelty import NoveltyCheck, NoveltyChecker
from llm_gent.core.training import StarPreferencePair, StarRatedItem


pytestmark = pytest.mark.unit


def _novelty_config(threshold: float = 0.85, mode: str | None = None) -> DotDict:
    """Create novelty config for tests."""
    config = DotDict({"similarity": {"threshold": threshold}})
    if mode:
        config["mode"] = mode
    return config


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
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        assert checker.has_embedder is True

        mock_learn_trait.has_embedder = False
        assert checker.has_embedder is False

    def test_check_novel_no_existing_jokes(self, mock_logger, mock_learn_trait):
        """Check returns novel when no existing jokes found."""
        mock_learn_trait.recall.return_value = []
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())

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

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
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

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 0.92
        assert result.similar_joke == "Very similar joke"

    def test_check_fails_closed_on_error(self, mock_logger, mock_learn_trait):
        """Check returns not novel on error (fail closed)."""
        mock_learn_trait.recall.side_effect = Exception("Database error")

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is False
        assert result.max_similarity == 1.0

    def test_check_novel_when_table_not_exists(self, mock_logger, mock_learn_trait):
        """Check returns novel when metadata table doesn't exist (lazy creation)."""
        # Simulate PostgreSQL "relation does not exist" error
        mock_learn_trait.recall.side_effect = Exception(
            'relation "playground.agent_jokester_training" does not exist'
        )

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        # Should treat as novel (no existing jokes), not fail closed
        assert result.is_novel is True
        assert result.max_similarity == 0.0
        assert result.similar_joke is None

    def test_check_novel_when_table_not_exists_sqlite(self, mock_logger, mock_learn_trait):
        """Check returns novel on SQLite 'no such table' error."""
        mock_learn_trait.recall.side_effect = Exception("no such table: agent_jokester_training")

        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())
        result = checker.check("Why did the chicken cross the road?")

        assert result.is_novel is True
        assert result.max_similarity == 0.0

    def test_check_rejects_empty_text(self, mock_logger, mock_learn_trait):
        """Check rejects empty joke text without calling embedder."""
        checker = NoveltyChecker(mock_logger, mock_learn_trait, _novelty_config())

        # Empty string
        result = checker.check("")
        assert result.is_novel is False
        assert result.max_similarity == 1.0
        mock_learn_trait.recall.assert_not_called()

        # Whitespace only
        result = checker.check("   ")
        assert result.is_novel is False
        mock_learn_trait.recall.assert_not_called()


class TestPreferencePair:
    """Tests for preference pairing types."""

    def test_rated_item(self):
        """StarRatedItem stores item data."""
        item = StarRatedItem(id=1, content="Funny joke", score=4)
        assert item.id == 1
        assert item.content == "Funny joke"
        assert item.score == 4

    def test_preference_pair_margin(self):
        """PreferencePair margin is score difference."""
        chosen = StarRatedItem(id=1, content="Great joke", score=5)
        rejected = StarRatedItem(id=2, content="Bad joke", score=1)
        pair = StarPreferencePair(chosen=chosen, rejected=rejected)

        assert pair.chosen.score - pair.rejected.score == 4

    def test_preference_pair_zero_margin(self):
        """PreferencePair with same score has zero margin."""
        chosen = StarRatedItem(id=1, content="Joke A", score=3)
        rejected = StarRatedItem(id=2, content="Joke B", score=3)
        pair = StarPreferencePair(chosen=chosen, rejected=rejected)

        assert pair.chosen.score - pair.rejected.score == 0


class TestJokeHistory:
    """Tests for JokeHistory."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    def test_contains_empty_history(self, mock_logger):
        """Contains returns False for empty history."""
        history = JokeHistory(mock_logger)
        assert history.contains("any joke") is False

    def test_contains_after_record(self, mock_logger):
        """Contains returns True after recording joke."""
        history = JokeHistory(mock_logger)
        history.record("Why did the chicken cross the road?")
        assert history.contains("Why did the chicken cross the road?") is True
        assert history.contains("Different joke") is False

    def test_contains_bumps_frequency(self, mock_logger):
        """Recording same joke multiple times bumps frequency."""
        history = JokeHistory(mock_logger)
        history.record("Same joke")
        history.record("Same joke")
        history.record("Same joke")
        # Frequency should be 3, check via _frequency (internal but useful for test)
        assert history._frequency["Same joke"] == 3
