import subtitle_renderer as sr


def test_build_style_config_uses_named_preset():
    style = sr.build_style_config({"preset": "youtube_moderno"})

    assert style.preset == "youtube_moderno"
    assert style.fontsize == 70
    assert style.max_chars_per_line == 26


def test_prepare_segments_applies_preview_limit_and_wraps_lines():
    style = sr.build_style_config(
        {
            "preset": "documental_limpio",
            "max_chars_per_line": 12,
            "max_lines": 2,
        }
    )
    segments = [
        {
            "start": 0.0,
            "end": 8.0,
            "text": "Este texto es demasiado largo para mantenerse en una sola linea sin wrap.",
        },
        {
            "start": 40.0,
            "end": 45.0,
            "text": "Debe quedar fuera del preview.",
        },
    ]

    prepared = sr._prepare_segments_for_render(segments, style, preview_duration_sec=30.0)

    assert len(prepared) == 1
    assert prepared[0]["end"] == 8.0
    assert "\n" in prepared[0]["text"]


def test_generate_drawtext_filter_includes_box_and_preview_clamp():
    filter_str = sr.generate_ffmpeg_drawtext_filter(
        [{"start": 0.0, "end": 8.0, "text": "Linea uno para preview"}],
        video_width=1920,
        video_height=1080,
        style_config={
            "preset": "caja_negra_broadcast",
            "box_enabled": True,
            "box_opacity": 0.75,
        },
        preview_duration_sec=3.0,
    )

    assert "box=1" in filter_str
    assert "boxcolor=0x000000@0.75" in filter_str
    assert "between(t,0.000,3.000)" in filter_str