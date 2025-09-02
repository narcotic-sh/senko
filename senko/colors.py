import numpy as np
import colour
from colour.models import RGB_COLOURSPACE_sRGB
from colour import CCS_ILLUMINANTS

GOLDEN_ANGLE = 137.508  # degrees
L_MIN = 0.325           # Minimum lightness
L_MAX = 0.85            # Maximum lightness
FIXED_CHROMA = 0.15     # Fixed chroma value
W = 0.5                 # Parameter for weighting segment count vs. speaking time: 0 (speaking time emphasis) to 1 (segment count emphasis)
D65 = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
L_a = 63.66             # Adapting luminance for H-K effect

#######################################################################
#   Generate single set of speaker colors
#######################################################################

def generate_speaker_colors(segments_data, i_advance):
    # Calculate total speaking time and segment counts per speaker
    speaking_times = {}
    segment_counts = {}
    for segment in segments_data:
        duration = segment['end'] - segment['start']
        speaker = segment['speaker']
        speaking_times[speaker] = speaking_times.get(speaker, 0) + duration
        segment_counts[speaker] = segment_counts.get(speaker, 0) + 1

    # Calculate total speaking time and segments across all speakers
    total_speaking_time = sum(speaking_times.values())
    total_segments = sum(segment_counts.values())

    # Rationalize speaking times and segment counts
    r_speaking_times = {speaker: time / total_speaking_time if total_speaking_time > 0 else 0
                        for speaker, time in speaking_times.items()}
    r_segment_counts = {speaker: count / total_segments if total_segments > 0 else 0
                        for speaker, count in segment_counts.items()}

    # Calculate lightness scores
    lightness_scores = {}
    for speaker in speaking_times:
        num_segments = r_segment_counts.get(speaker, 0)
        speaking_time = r_speaking_times.get(speaker, 0)
        score = W * num_segments + (1 - W) * (1 - speaking_time)
        lightness_scores[speaker] = score

    # Normalize lightness scores to [L_MIN, L_MAX]
    scores = list(lightness_scores.values())
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    lightness = {}
    power = 0.33  # power for Stevens's power law ; dunno if this is theoretically correct, but yields good results (point is to add non-linearity somewhere in here)
    for speaker, score in lightness_scores.items():
        if max_score == min_score:
            normalized = 0.5
        else:
            normalized = ((score - min_score) / (max_score - min_score)) ** power
        lightness[speaker] = L_MIN + normalized * (L_MAX - L_MIN)

    # Sort speakers by speaking time (descending)
    speakers = sorted(speaking_times, key=speaking_times.get, reverse=True)

    # Assign colors with H-K adjustment
    speaker_colors = {}
    for i, speaker in enumerate(speakers):
        L_original = lightness[speaker]
        H = ((i+i_advance) * GOLDEN_ANGLE) % 360
        C = FIXED_CHROMA

        # Compute Γ for the original OKLCH color
        Gamma = compute_hk_factor(L_original, C, H)

        # Adjust lightness
        L_adjusted = L_original / Gamma
        L_adjusted = max(0.0, min(1.0, L_adjusted))  # Clamp to [0, 1]

        # Convert adjusted OKLCH to hex
        hex_color = oklch_to_hex(L_adjusted, C, H)
        speaker_colors[speaker] = hex_color

    return speaker_colors

#######################################################################
#   Helpers
#######################################################################

def oklch_to_hex(L, C, H):
    """Convert OKLCH color to hex code."""
    JCh = np.array([L, C, H])
    Jab = colour.models.JCh_to_Jab(JCh)
    XYZ = colour.models.oklab.Oklab_to_XYZ(Jab)
    RGB = colour.XYZ_to_RGB(XYZ, RGB_COLOURSPACE_sRGB, illuminant=D65, apply_cctf_encoding=True)
    RGB = np.clip(RGB, 0, 1)  # Ensure RGB values are within [0, 1]
    return colour.notation.RGB_to_HEX(RGB)

def compute_hk_factor(L, C, H):
    """Compute the Helmholtz-Kohlrausch factor Γ for the given OKLCH color."""
    # Convert OKLCH to XYZ
    JCh = np.array([L, C, H])
    Jab = colour.models.JCh_to_Jab(JCh)
    XYZ = colour.models.oklab.Oklab_to_XYZ(Jab)

    # Compute CIE uv chromaticity for the color
    denominator = XYZ[0] + 15 * XYZ[1] + 3 * XYZ[2]
    if denominator == 0:
        uv = np.array([0.0, 0.0])
    else:
        uv = np.array([4 * XYZ[0] / denominator, 9 * XYZ[1] / denominator])

    # Reference white: D65
    D65_XYZ = np.array([95.047, 100.0, 108.883]) / 100  # Normalize to [0, 1]
    denominator_w = D65_XYZ[0] + 15 * D65_XYZ[1] + 3 * D65_XYZ[2]
    uv_c = np.array([4 * D65_XYZ[0] / denominator_w, 9 * D65_XYZ[1] / denominator_w])

    # Compute Γ using Nayatani's method
    Gamma = colour.HelmholtzKohlrausch_effect_object_Nayatani1997(
        uv, uv_c, L_a, method='VCC'
    )
    return Gamma
