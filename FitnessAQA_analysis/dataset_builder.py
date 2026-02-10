"""
Dataset Builder for LLM Fine-Tuning on Fitness Biomechanics

Generates instruction-tuning datasets from encoded pose time series.
Outputs Alpaca/ShareGPT-compatible formats for Unsloth training.

Produces diverse QA pairs covering:
  - Exercise scoring and quality assessment
  - Specific rep analysis
  - Form correction recommendations
  - Biomechanical metric interpretation
  - Phase-by-phase breakdown
"""

import json
import os
import random
import numpy as np
from typing import List, Dict, Optional
from time_series_encoder import TimeSeriesEncoder, EncoderConfig


# ─────────────── Instruction Templates ───────────────

SYSTEM_PROMPT = (
    "You are FitnessGPT, an expert AI biomechanics coach. You analyze exercise "
    "form using pose estimation data encoded as structured tokens. Each token sequence "
    "represents joint positions, angles, velocities, and movement phases captured at "
    "30fps. You provide precise, actionable coaching feedback grounded in the data."
)

# Template categories for diverse training data
INSTRUCTION_TEMPLATES = {
    "overall_assessment": [
        "Analyze the following pull-up exercise data and provide an overall quality assessment.\n\nPose Data:\n{summary}\n\nQuality Indicators: {quality}",
        "Review this pull-up performance. What is the overall form quality?\n\n{summary}\n\n{quality}",
        "Based on the biomechanical data below, rate this pull-up set and explain your reasoning.\n\n{summary}",
    ],
    "rep_analysis": [
        "Analyze rep {rep_num} of this pull-up set. What was the quality of this repetition?\n\nFull Data:\n{summary}\n\nRep {rep_num} Data:\n{rep_detail}",
        "How was the {rep_num}th repetition? Provide specific feedback.\n\n{summary}\n\nRep detail: {rep_detail}",
    ],
    "form_correction": [
        "What form corrections should this person focus on based on their pull-up data?\n\n{summary}\n\n{quality}",
        "Identify the biggest form issues in this pull-up set and recommend fixes.\n\n{summary}",
        "As a biomechanics coach, what would you tell this person to improve?\n\n{summary}\n\n{quality}",
    ],
    "metric_interpretation": [
        "Explain what these biomechanical metrics mean for this pull-up performance:\n\n{summary}",
        "Interpret the following exercise metrics and explain their significance:\n\n{summary}",
        "What do the efficiency and ROM numbers tell us about this athlete's pull-ups?\n\n{summary}",
    ],
    "phase_analysis": [
        "Analyze the movement phases in this pull-up set.\n\n{summary}\n\nPhase tokens from the sequence: {phase_info}",
        "Break down the concentric and eccentric phases of these pull-ups.\n\n{summary}\n\n{phase_info}",
    ],
    "comparison": [
        "Compare the first rep to the last rep. Is there fatigue indication?\n\n{summary}\n\nFirst rep: {first_rep}\nLast rep: {last_rep}",
    ],
    "tokenized_analysis": [
        "Analyze this encoded pose sequence and provide coaching feedback.\n\nSequence:\n{frame_tokens}\n\nSummary:\n{summary}",
        "Given the following tokenized biomechanical data, assess the exercise quality.\n\n{frame_tokens}\n\nMetrics:\n{summary}",
    ],
    "symmetry_analysis": [
        "Analyze the left-right symmetry of this pull-up performance.\n\n{summary}\n\nSymmetry data: {symmetry}",
        "Is this person pulling evenly with both arms? Analyze the symmetry.\n\n{summary}\n\n{symmetry}",
    ],
}


class ResponseGenerator:
    """Generates contextual responses based on extracted features."""
    
    @staticmethod
    def overall_assessment(features: Dict, quality: str) -> str:
        s = features["summary"]
        reps = features["reps"]
        
        # Determine quality tier
        eff = s["mechanical_efficiency_pct"]
        rom = s["useful_work_rom"]
        
        if eff > 80 and rom > 0.7:
            tier = "excellent"
            grade = "A"
        elif eff > 60 and rom > 0.5:
            tier = "good"
            grade = "B"
        elif eff > 40 and rom > 0.3:
            tier = "fair"
            grade = "C"
        else:
            tier = "needs improvement"
            grade = "D"
        
        response = f"**Overall Assessment: {tier.title()} ({grade})**\n\n"
        response += f"This pull-up set consisted of {len(reps)} repetitions over {s['duration_sec']:.1f} seconds.\n\n"
        
        response += "**Key Metrics:**\n"
        response += f"- Mechanical Efficiency: {eff:.1f}% — "
        if eff > 70:
            response += "Very efficient movement pattern with minimal wasted energy.\n"
        elif eff > 50:
            response += "Moderate efficiency. Some energy lost to lateral movement or jerky motion.\n"
        else:
            response += "Low efficiency. Significant energy wasted on non-productive movement.\n"
        
        response += f"- Range of Motion: {rom:.3f} body units — "
        if rom > 0.7:
            response += "Full ROM achieved. Chin clearly over bar with full extension at bottom.\n"
        elif rom > 0.4:
            response += "Partial ROM. Consider pulling higher or extending further at the bottom.\n"
        else:
            response += "Limited ROM. Focus on achieving full range of motion.\n"
        
        response += f"- Efficiency Signal: {s['efficiency_db']:.1f} dB — "
        if s['efficiency_db'] > 10:
            response += "Strong signal-to-noise ratio. Clean, controlled movement.\n"
        elif s['efficiency_db'] > 0:
            response += "Acceptable signal-to-noise. Some unnecessary movement present.\n"
        else:
            response += "Poor signal-to-noise. Significant body swing or jerky motion.\n"
        
        response += f"\n**Movement Quality:** {quality}\n"
        
        # Rep-by-rep consistency
        if len(reps) > 1:
            rom_values = [r["rom"] for r in reps]
            rom_std = np.std(rom_values)
            response += f"\n**Consistency:** ROM standard deviation across reps: {rom_std:.3f}. "
            if rom_std < 0.05:
                response += "Very consistent repetitions — excellent motor control."
            elif rom_std < 0.15:
                response += "Moderately consistent. Some variation between reps is normal."
            else:
                response += "High variation between reps. May indicate fatigue or inconsistent technique."
        
        return response
    
    @staticmethod
    def rep_analysis(features: Dict, rep_num: int) -> str:
        reps = features["reps"]
        if rep_num < 1 or rep_num > len(reps):
            return f"Rep {rep_num} not found in the data. Total reps detected: {len(reps)}."
        
        rep = reps[rep_num - 1]
        response = f"**Rep {rep_num} Analysis:**\n\n"
        response += f"- Frame range: {rep['start_frame']} to {rep['end_frame']} "
        response += f"({rep['end_frame'] - rep['start_frame']} frames, "
        response += f"{(rep['end_frame'] - rep['start_frame']) / 30:.1f}s)\n"
        response += f"- Peak height: {rep['peak_height']:.3f} (normalized)\n"
        response += f"- Range of Motion: {rep['rom']:.3f}\n\n"
        
        if rep["rom"] > 0.7:
            response += "This was a full-range repetition with excellent ROM. "
        elif rep["rom"] > 0.4:
            response += "This rep had moderate ROM — could improve by pulling higher. "
        else:
            response += "Partial rep — the ROM was limited. Focus on full range of motion. "
        
        # Compare to average
        if len(reps) > 1:
            avg_rom = np.mean([r["rom"] for r in reps])
            if rep["rom"] > avg_rom * 1.1:
                response += "This rep was above average for the set."
            elif rep["rom"] < avg_rom * 0.9:
                response += "This rep was below average — possible fatigue."
            else:
                response += "This rep was consistent with the set average."
        
        return response
    
    @staticmethod
    def form_correction(features: Dict) -> str:
        s = features["summary"]
        sym = features["symmetry"]
        
        corrections = []
        
        # ROM issue
        if s["useful_work_rom"] < 0.5:
            corrections.append(
                "**Increase Range of Motion:** Your ROM is limited. Focus on full extension "
                "at the bottom (dead hang with straight arms) and pulling until chin is "
                "clearly over the bar. Use assisted pull-ups if needed to achieve full ROM."
            )
        
        # Stability issue
        if s["horizontal_energy"] > 0.3:
            corrections.append(
                "**Reduce Body Swing:** Significant lateral/horizontal movement detected. "
                "Engage your core throughout the movement. Avoid kipping unless intentional. "
                "Try pausing at the bottom of each rep to eliminate momentum."
            )
        
        # Smoothness issue
        if s["jerk_energy"] > 1.5:
            corrections.append(
                "**Control the Movement:** High jerk values indicate choppy motion. "
                "Focus on smooth, controlled pulling and lowering. Use a 2-1-3 tempo: "
                "2 seconds up, 1 second hold at top, 3 seconds controlled descent."
            )
        
        # Efficiency issue
        if s["mechanical_efficiency_pct"] < 50:
            corrections.append(
                "**Improve Movement Efficiency:** Your mechanical efficiency is low, meaning "
                "a lot of energy is going into non-productive movement. Focus on pulling "
                "straight up and down, minimizing body sway, and using a controlled tempo."
            )
        
        # Symmetry issue
        if sym:
            avg_asym = np.mean(list(sym.values()))
            if avg_asym > 15:
                corrections.append(
                    "**Address Asymmetry:** Significant left-right imbalance detected. "
                    "This could indicate strength imbalance or a compensatory pattern. "
                    "Try single-arm hangs and eccentric-focused work on the weaker side."
                )
        
        if not corrections:
            return (
                "**Form looks solid!** No major corrections needed. To continue improving:\n"
                "- Add weight progressively\n"
                "- Experiment with grip variations (wide, narrow, neutral)\n"
                "- Incorporate tempo work (slow eccentrics)\n"
                "- Track your efficiency metrics over time"
            )
        
        return "**Form Corrections (Priority Order):**\n\n" + "\n\n".join(
            f"{i+1}. {c}" for i, c in enumerate(corrections)
        )
    
    @staticmethod
    def metric_interpretation(features: Dict) -> str:
        s = features["summary"]
        
        response = "**Metric Interpretation:**\n\n"
        
        response += f"**Mechanical Efficiency ({s['mechanical_efficiency_pct']:.1f}%):** "
        response += (
            "This measures what percentage of your total energy expenditure goes into "
            "productive vertical movement. Higher is better. Elite pull-up form typically "
            f"shows >80%. Your {s['mechanical_efficiency_pct']:.1f}% "
        )
        if s['mechanical_efficiency_pct'] > 80:
            response += "is excellent — very little wasted energy.\n\n"
        elif s['mechanical_efficiency_pct'] > 60:
            response += "is good but has room for improvement.\n\n"
        else:
            response += "suggests significant energy loss to non-productive movement patterns.\n\n"
        
        response += f"**Efficiency Signal ({s['efficiency_db']:.1f} dB):** "
        response += (
            "This is a signal-to-noise ratio comparing useful work (vertical ROM) to "
            "wasted energy (horizontal movement + jerk). Positive dB means more signal "
            "than noise. "
        )
        if s['efficiency_db'] > 10:
            response += "Your positive value indicates clean technique.\n\n"
        else:
            response += "Consider reducing body swing and focusing on smooth control.\n\n"
        
        response += f"**Horizontal Energy ({s['horizontal_energy']:.4f}):** "
        response += (
            "Measures lateral body movement. In a strict pull-up, this should be "
            "near zero. Higher values indicate swinging or kipping patterns.\n\n"
        )
        
        response += f"**Jerk Energy ({s['jerk_energy']:.4f}):** "
        response += (
            "Jerk is the rate of change of acceleration — it measures movement smoothness. "
            "Lower values indicate smoother, more controlled motion. High jerk suggests "
            "sudden direction changes or compensatory movements."
        )
        
        return response
    
    @staticmethod
    def phase_analysis(features: Dict) -> str:
        phases = features["phases"]
        reps = features["reps"]
        
        response = "**Movement Phase Analysis:**\n\n"
        
        if not phases:
            return response + "Insufficient phase data detected. The movement may be too short for phase analysis."
        
        # Count phase types
        phase_counts = {}
        total_frames = 0
        for p in phases:
            phase_counts.setdefault(p["type"], {"count": 0, "total_frames": 0})
            phase_counts[p["type"]]["count"] += 1
            phase_counts[p["type"]]["total_frames"] += p["duration_frames"]
            total_frames += p["duration_frames"]
        
        for ptype, info in phase_counts.items():
            pct = info["total_frames"] / total_frames * 100 if total_frames > 0 else 0
            response += f"- **{ptype.title()}** phases: {info['count']} occurrences, "
            response += f"{info['total_frames']} frames ({pct:.1f}% of movement)\n"
        
        response += "\n"
        
        # Eccentric vs concentric timing
        ecc = phase_counts.get("eccentric", {"total_frames": 0})["total_frames"]
        conc = phase_counts.get("concentric", {"total_frames": 0})["total_frames"]
        
        if conc > 0:
            ratio = ecc / conc
            response += f"**Eccentric:Concentric ratio: {ratio:.2f}**\n"
            if ratio > 1.5:
                response += "Good eccentric control — you're lowering slower than pulling up, "
                response += "which is ideal for strength and muscle development.\n"
            elif ratio > 0.8:
                response += "Roughly equal concentric and eccentric phases. "
                response += "Consider slowing down the lowering phase for more time under tension.\n"
            else:
                response += "Fast eccentric (lowering) phase. Slow down the descent to "
                response += "reduce injury risk and increase strength gains.\n"
        
        return response
    
    @staticmethod
    def fatigue_analysis(features: Dict) -> str:
        reps = features["reps"]
        if len(reps) < 2:
            return "Need at least 2 reps to analyze fatigue."
        
        first = reps[0]
        last = reps[-1]
        
        response = "**Fatigue Analysis (First vs Last Rep):**\n\n"
        response += f"- First rep ROM: {first['rom']:.3f}, Peak: {first['peak_height']:.3f}\n"
        response += f"- Last rep ROM: {last['rom']:.3f}, Peak: {last['peak_height']:.3f}\n"
        
        rom_drop = (first['rom'] - last['rom']) / first['rom'] * 100 if first['rom'] > 0 else 0
        response += f"- ROM decline: {rom_drop:.1f}%\n\n"
        
        if rom_drop > 20:
            response += (
                "Significant fatigue detected. ROM dropped substantially from first to last rep. "
                "Consider reducing rep count or taking longer rest periods. "
                "Quality over quantity — stop when ROM drops >15%."
            )
        elif rom_drop > 10:
            response += (
                "Moderate fatigue. Some ROM decline is normal towards the end of a set. "
                "You're managing fatigue reasonably well."
            )
        elif rom_drop > 0:
            response += (
                "Minimal fatigue indication. Excellent endurance and motor control "
                "maintained throughout the set."
            )
        else:
            response += (
                "ROM actually improved across the set — you may have been warming up "
                "or the first rep was conservative. No fatigue concerns."
            )
        
        return response


class DatasetBuilder:
    """Builds instruction-tuning datasets from encoded exercise data."""
    
    def __init__(self, encoder: Optional[TimeSeriesEncoder] = None):
        self.encoder = encoder or TimeSeriesEncoder()
        self.response_gen = ResponseGenerator()
    
    def build_from_json(self, json_path: str, exercise_type: str = "pullup",
                        stride: int = 3, augment: bool = True) -> List[Dict]:
        """
        Build training examples from a single JSON file.
        
        Returns list of instruction-tuning examples in Alpaca format:
        [{"instruction": ..., "input": ..., "output": ...}, ...]
        """
        encoded = self.encoder.encode_from_json(json_path, stride=stride)
        features = encoded["features"]
        
        examples = []
        
        # ── Overall Assessment ──
        for template in INSTRUCTION_TEMPLATES["overall_assessment"]:
            instruction = template.format(
                summary=encoded["summary_tokens"],
                quality=encoded["quality_tokens"],
            )
            response = self.response_gen.overall_assessment(features, encoded["quality_tokens"])
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
        
        # ── Rep Analysis ──
        for rep in features["reps"]:
            rep_num = rep["rep_number"]
            rep_detail = (
                f"start_frame: {rep['start_frame']}, end_frame: {rep['end_frame']}, "
                f"peak_height: {rep['peak_height']:.3f}, rom: {rep['rom']:.3f}"
            )
            
            for template in INSTRUCTION_TEMPLATES["rep_analysis"]:
                instruction = template.format(
                    rep_num=rep_num,
                    summary=encoded["summary_tokens"],
                    rep_detail=rep_detail,
                )
                response = self.response_gen.rep_analysis(features, rep_num)
                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                })
        
        # ── Form Correction ──
        for template in INSTRUCTION_TEMPLATES["form_correction"]:
            instruction = template.format(
                summary=encoded["summary_tokens"],
                quality=encoded["quality_tokens"],
            )
            response = self.response_gen.form_correction(features)
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
        
        # ── Metric Interpretation ──
        for template in INSTRUCTION_TEMPLATES["metric_interpretation"]:
            instruction = template.format(summary=encoded["summary_tokens"])
            response = self.response_gen.metric_interpretation(features)
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
        
        # ── Phase Analysis ──
        phase_info = "; ".join(
            f"{p['type']}({p['start_frame']}-{p['end_frame']})"
            for p in features["phases"][:10]  # Limit for token budget
        )
        for template in INSTRUCTION_TEMPLATES["phase_analysis"]:
            instruction = template.format(
                summary=encoded["summary_tokens"],
                phase_info=phase_info,
            )
            response = self.response_gen.phase_analysis(features)
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
        
        # ── Fatigue / Comparison ──
        if len(features["reps"]) >= 2:
            first_rep = features["reps"][0]
            last_rep = features["reps"][-1]
            first_str = f"rom: {first_rep['rom']:.3f}, peak: {first_rep['peak_height']:.3f}"
            last_str = f"rom: {last_rep['rom']:.3f}, peak: {last_rep['peak_height']:.3f}"
            
            for template in INSTRUCTION_TEMPLATES["comparison"]:
                instruction = template.format(
                    summary=encoded["summary_tokens"],
                    first_rep=first_str,
                    last_rep=last_str,
                )
                response = self.response_gen.fatigue_analysis(features)
                examples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": response,
                })
        
        # ── Tokenized Sequence Analysis ──
        # Use truncated frame tokens to stay within context window
        max_token_chars = 2000
        truncated_frames = encoded["frame_tokens"][:max_token_chars]
        
        for template in INSTRUCTION_TEMPLATES["tokenized_analysis"]:
            instruction = template.format(
                frame_tokens=truncated_frames,
                summary=encoded["summary_tokens"],
            )
            response = self.response_gen.overall_assessment(features, encoded["quality_tokens"])
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
        
        # ── Symmetry Analysis ──
        sym_str = ", ".join(f"{k}: {v:.1f}°" for k, v in features["symmetry"].items())
        for template in INSTRUCTION_TEMPLATES["symmetry_analysis"]:
            instruction = template.format(
                summary=encoded["summary_tokens"],
                symmetry=sym_str,
            )
            # Simple symmetry response
            sym_vals = list(features["symmetry"].values())
            avg_asym = np.mean(sym_vals) if sym_vals else 0
            if avg_asym < 5:
                sym_response = (
                    "**Excellent symmetry.** Left and right sides show very similar "
                    "joint angles throughout the movement. Average asymmetry: "
                    f"{avg_asym:.1f}°. This indicates balanced strength and good "
                    "neuromuscular coordination."
                )
            elif avg_asym < 15:
                sym_response = (
                    f"**Moderate symmetry.** Average angle difference: {avg_asym:.1f}°. "
                    "Some left-right imbalance is present but within acceptable range. "
                    "Monitor this over time and consider unilateral training if the "
                    "asymmetry increases."
                )
            else:
                sym_response = (
                    f"**Significant asymmetry detected.** Average angle difference: "
                    f"{avg_asym:.1f}°. This could indicate a strength imbalance, "
                    "mobility restriction, or compensatory movement pattern. "
                    "Recommendation: Incorporate single-arm work and address any "
                    "mobility limitations."
                )
            examples.append({
                "instruction": instruction,
                "input": "",
                "output": sym_response,
            })
        
        # ── Data Augmentation ──
        if augment:
            augmented = self._augment_examples(examples)
            examples.extend(augmented)
        
        # Add system prompt to all examples
        for ex in examples:
            ex["system"] = SYSTEM_PROMPT
        
        return examples
    
    def _augment_examples(self, examples: List[Dict]) -> List[Dict]:
        """Create augmented variants of training examples."""
        augmented = []
        
        for ex in examples[:5]:  # Augment a subset
            # Rephrase style: more casual
            casual = {
                "instruction": "Hey, can you look at my pull-up data and tell me how I did?\n\n" + 
                              ex["instruction"].split("\n\n", 1)[-1] if "\n\n" in ex["instruction"] else ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            }
            augmented.append(casual)
            
            # Rephrase style: concise request
            concise = {
                "instruction": "Quick analysis:\n" + 
                              ex["instruction"].split("\n\n", 1)[-1] if "\n\n" in ex["instruction"] else ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            }
            augmented.append(concise)
        
        return augmented
    
    def build_from_directory(self, data_dir: str, stride: int = 3) -> List[Dict]:
        """Build dataset from all JSON files in a directory."""
        all_examples = []
        
        for fname in os.listdir(data_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(data_dir, fname)
                try:
                    examples = self.build_from_json(fpath, stride=stride)
                    all_examples.extend(examples)
                    print(f"  ✓ {fname}: {len(examples)} examples")
                except Exception as e:
                    print(f"  ✗ {fname}: {e}")
        
        return all_examples
    
    def save_dataset(self, examples: List[Dict], output_path: str,
                     format: str = "alpaca"):
        """
        Save dataset in the specified format.
        
        Formats:
            alpaca: [{"instruction": ..., "input": ..., "output": ...}]
            sharegpt: [{"conversations": [{"from": "system",...}, {"from": "human",...}, {"from": "gpt",...}]}]
        """
        if format == "sharegpt":
            converted = []
            for ex in examples:
                conv = {
                    "conversations": [
                        {"from": "system", "value": ex.get("system", SYSTEM_PROMPT)},
                        {"from": "human", "value": ex["instruction"]},
                        {"from": "gpt", "value": ex["output"]},
                    ]
                }
                converted.append(conv)
            data = converted
        else:
            data = examples
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} examples to {output_path} ({format} format)")
    
    def get_dataset_stats(self, examples: List[Dict]) -> Dict:
        """Compute dataset statistics."""
        instruction_lengths = [len(ex["instruction"].split()) for ex in examples]
        output_lengths = [len(ex["output"].split()) for ex in examples]
        
        return {
            "total_examples": len(examples),
            "avg_instruction_words": np.mean(instruction_lengths),
            "avg_output_words": np.mean(output_lengths),
            "max_instruction_words": max(instruction_lengths),
            "max_output_words": max(output_lengths),
            "total_words": sum(instruction_lengths) + sum(output_lengths),
        }


# ───────────────── Standalone Usage ─────────────────

if __name__ == "__main__":
    print("=== Dataset Builder ===\n")
    
    builder = DatasetBuilder()
    
    # Build from single file
    examples = builder.build_from_json("./data/pullup_analysis.json", stride=5)
    
    stats = builder.get_dataset_stats(examples)
    print(f"Dataset Stats: {json.dumps(stats, indent=2)}")
    
    # Save in both formats
    builder.save_dataset(examples, "./outputs/train_alpaca.json", format="alpaca")
    builder.save_dataset(examples, "./outputs/train_sharegpt.json", format="sharegpt")
    
    # Print sample
    print("\n--- Sample Example ---")
    sample = examples[0]
    print(f"Instruction (first 200 chars): {sample['instruction'][:200]}...")
    print(f"Output (first 300 chars): {sample['output'][:300]}...")
