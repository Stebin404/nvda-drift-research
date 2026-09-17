from evaluation.detector_metrics import evaluate_detector

ground_truth = [637, 663, 728]

# Regression test for the greedy nearest-match bugfix -- pinned to
# one_sided=False so it isolates "nearest, not first-in-list" behavior
# from the causal-window feature added alongside it.
result = evaluate_detector([655], ground_truth, tolerance=30, one_sided=False)
print("Alarm at 655 vs ground truth [637, 663, 728] (symmetric):")
print(result)
assert result["avg_delay"] == 8, "Should match nearest point (663), not first-in-list (637)"

result2 = evaluate_detector([640, 660, 730], ground_truth, tolerance=30, one_sided=False)
print("\nAlarms [640, 660, 730] vs ground truth [637, 663, 728] (symmetric):")
print(result2)
assert result2["matched_truth"] == 3
assert result2["matched_alarms"] == 3

# New tests for one_sided=True (default): alarm must occur ON OR AFTER
# the event to count as a detection of it.
result3 = evaluate_detector([655], ground_truth, tolerance=30, one_sided=True)
print("\nAlarm at 655 vs ground truth [637, 663, 728] (one-sided, default):")
print(result3)
assert result3["matched_truth"] == 1
assert result3["avg_delay"] == 18, "One-sided mode should match 637 (alarm after event), not 663 (alarm before)"

result4 = evaluate_detector([600], [620], tolerance=30, one_sided=True)
print("\nAlarm at 600 vs ground truth [620] (one-sided, alarm precedes event):")
print(result4)
assert result4["matched_truth"] == 0, "An alarm before the event must not count as a detection in one-sided mode"

print("\nALL TESTS PASSED")