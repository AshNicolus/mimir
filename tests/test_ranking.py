"""The Beta-posterior confidence estimator and its incomplete-beta core."""

from mimir.ranking import beta_lower_bound, incomplete_beta


def test_incomplete_beta_matches_analytic_cases():
    assert abs(incomplete_beta(1, 1, 0.3) - 0.3) < 1e-9  # uniform CDF
    assert abs(incomplete_beta(2, 1, 0.4) - 0.16) < 1e-9  # x^2
    assert abs(incomplete_beta(1, 2, 0.4) - 0.64) < 1e-9  # 1 - (1-x)^2
    assert abs(incomplete_beta(3, 3, 0.5) - 0.5) < 1e-9  # symmetric


def test_incomplete_beta_saturates_outside_unit_interval():
    assert incomplete_beta(2, 3, 0.0) == 0.0
    assert incomplete_beta(2, 3, 1.0) == 1.0


def test_beta_lower_bound_is_the_tail_quantile():
    for successes, failures in [(1, 0), (9, 1), (5, 5), (250, 0), (0.5, 0.2)]:
        bound = beta_lower_bound(successes, failures)
        cdf = incomplete_beta(0.5 + successes, 0.5 + failures, bound)
        assert abs(cdf - 0.05) < 1e-3


def test_beta_lower_bound_rewards_more_evidence():
    assert beta_lower_bound(5, 0) < beta_lower_bound(50, 0)


def test_beta_lower_bound_penalizes_failures():
    assert beta_lower_bound(5, 0) > beta_lower_bound(5, 5)


def test_beta_lower_bound_is_zero_without_successes():
    assert beta_lower_bound(0, 3) == 0.0


def test_lighter_weighted_evidence_ranks_below_equal_raw_evidence():
    # Same success rate, but downweighted evidence yields a lower bound, which
    # is how relevance and recency weighting change the ranking.
    assert beta_lower_bound(2.0, 0.0) < beta_lower_bound(5.0, 0.0)
