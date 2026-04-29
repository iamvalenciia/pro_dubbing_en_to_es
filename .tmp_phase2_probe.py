import os
import json
import traceback
import main_ui

video_path = os.path.abspath('input/full_video_en.mp4')
phase_paths = main_ui._phase_artifact_paths(video_path)
print('VIDEO', video_path)
print('PHASE1_EXISTS', os.path.isfile(phase_paths['speaker_segments']))
print('PHASE2_EXISTS', os.path.isfile(phase_paths['translated_segments']))

spk_payload = main_ui._read_json_safe(phase_paths['speaker_segments'], {})
state_speakers = spk_payload.get('speakers') or []
print('STATE_SPEAKERS', len(state_speakers))

N = main_ui.N_SPK_MAX
voice_label = main_ui.STATIC_VOICE_LABELS[0] if main_ui.STATIC_VOICE_LABELS else None
voice_inputs = [voice_label for _ in range(N)]
label_inputs = ['' for _ in range(N)]

try:
    gen = main_ui.run_phase2_translation(video_path, state_speakers, phase_paths['speaker_segments'], *(voice_inputs + label_inputs))
    first = next(gen)
    print('FIRST_TEXT', first[0])
    print('FIRST_STATE_PHASE2', first[4])
    print('FIRST_FILE', first[5])
    last = first
    count = 1
    for item in gen:
        last = item
        count += 1
    print('YIELD_COUNT', count)
    print('LAST_TEXT', last[0])
    print('LAST_STATE_PHASE2', last[4])
    print('LAST_FILE', last[5])
except StopIteration:
    print('GEN_STOPPED_IMMEDIATELY')
except Exception as e:
    print('PY_ERROR', repr(e))
    traceback.print_exc()
