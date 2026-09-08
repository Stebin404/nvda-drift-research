from evaluation.detector_metrics import evaluate_detector

# Clustered ground truth, similar to the real May 2021 earnings/decoy
# situation -- two points close together, to confirm nearest-match
# (not first-in-list-match) is working correctly.
ground_truth = [637, 663, 728]

# An alarm at 655 is 18 away from 637, but only 8 away from 663.
# It should match 663, not 637.
result = evaluate_detector([655], ground_truth, tolerance=30)
print("Alarm at 655 vs ground truth [637, 663, 728]:")
print(result)
assert result["avg_delay"] == 8, "Should match nearest point (663), not first-in-list (637)"

# Multiple alarms, multiple truths, no double-counting
result2 = evaluate_detector([640, 660, 730], ground_truth, tolerance=30)
print("\nAlarms [640, 660, 730] vs ground truth [637, 663, 728]:")
print(result2)
assert result2["matched_truth"] == 3
assert result2["matched_alarms"] == 3

print("\nALL TESTS PASSED")