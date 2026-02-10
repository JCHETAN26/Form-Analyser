#!/bin/bash
# Quick Demo Script for Team Meeting
# Run this to show the full pipeline in action

echo "🎬 FitnessAQA CV Pipeline Demo"
echo "================================"
echo ""

# 1. Show what we have
echo "📁 Step 1: Available Test Videos"
echo "--------------------------------"
ls -1 test_videos/*.mp4
echo ""

# 2. Show existing results
echo "📊 Step 2: Generated Outputs (6 videos processed)"
echo "--------------------------------"
echo "Keypoints JSON files:"
ls -1 test_results/*_keypoints.json | wc -l
echo ""
echo "Analysis reports:"
ls -1 test_results/*_report.json | wc -l
echo ""
echo "Visualization videos:"
ls -1 test_results/*.mp4 | wc -l
echo ""
echo "Biomechanical graphs:"
ls -1 test_results/graphs/*.png | wc -l
echo ""

# 3. Show a sample analysis
echo "📋 Step 3: Sample Analysis - Perfect Pull-up"
echo "--------------------------------"
echo ""
echo "🤖 LLM Tokens:"
cat test_results/perfect_pullup_analysis_tokens.txt
echo ""
echo ""

echo "📈 Biomechanical Metrics:"
python3 -c "
import json
with open('test_results/perfect_pullup_analysis_report.json') as f:
    data = json.load(f)
print(f'  Reps: {data[\"rep_count\"]}')
print(f'  Avg Jerk: {data[\"avg_jerk\"]:.0f}')
print(f'  Avg Symmetry: {data[\"avg_symmetry\"]:.2f}')
print(f'  Total Frames: {len(data[\"frame_analysis\"])}')
"
echo ""

# 4. Show comparison
echo "⚖️  Step 4: Good vs Bad Form Comparison"
echo "--------------------------------"
echo ""
python3 << 'EOF'
import json

videos = [
    ("perfect_pullup", "Good form"),
    ("pullup_mistakes", "Bad form (kipping)"),
    ("perfect_form_squat", "Good squat"),
    ("common_mistakes", "Bad squat")
]

print(f"{'Video':<25} {'Expected':<25} {'Jerk':<12} {'Symmetry'}")
print("-" * 75)

for vid, desc in videos:
    try:
        with open(f'test_results/{vid}_analysis_report.json') as f:
            data = json.load(f)
        jerk = data['avg_jerk']
        sym = data['avg_symmetry']
        print(f"{vid:<25} {desc:<25} {jerk:<12.0f} {sym:.2f}")
    except:
        pass
EOF
echo ""

# 5. Show integration
echo "🔗 Step 5: Integration with FitnessGPT"
echo "--------------------------------"
echo "Training examples generated:"
python3 -c "import json; data = json.load(open('FitnessAQA_analysis/outputs/train_fitness_alpaca.json')); print(f'  {len(data)} examples ready for training')"
echo ""

# 6. Open key files
echo "🎥 Step 6: Opening Demo Files..."
echo "--------------------------------"
echo "Opening visualization video..."
open test_results/perfect_pullup_analysis.mp4 2>/dev/null || echo "  (Video player not available)"
echo ""
echo "Opening biomechanical graph..."
open test_results/graphs/perfect_pullup_graph.png 2>/dev/null || echo "  (Image viewer not available)"
echo ""

echo "✅ Demo Complete!"
echo ""
echo "📄 Key Documents:"
echo "  - VALIDATION_COMPLETE.md (results summary)"
echo "  - INTEGRATION_COMPLETE.md (Vishal handoff)"
echo "  - TEAM_MEETING_DEMO.md (this demo script)"
echo ""
echo "🚀 Ready for team presentation!"
