# Recording checklist

See `../../PLAN.md` Section 3 for the full rationale. This is the practical run sheet.

## Setup (once per session)

- Quiet, soft-furnished room (a closet with clothes hanging works well as a cheap vocal booth).
- Cardioid USB mic (Blue Yeti / AT2020USB+ or similar), ~1 fist-width from your mouth.
- Record at 24-bit, 44.1kHz or 48kHz. Keep levels between -24dB and -6dB.
- Save each session as **one long-form WAV file**, e.g. `harvard_session1.wav`,
  `conversational_session1.wav` — `voiceclone data prepare` chunks these automatically.
- Same mic, same room, same rough mic distance across sessions — consistency matters more than any
  single session being "perfect."

## Pilot first

Before recording everything below, record ~20-30 minutes (Harvard sentences is enough) and run the
pilot fine-tune (`PLAN.md` Section 5) end to end. Confirm the pipeline works and the output is
promising before investing hours in the full session list.

## Session list (aim for 3-5+ hours total, across multiple days)

1. **Harvard sentences** (~20-40 min) — phonetically balanced, standard TTS-dataset material.
   Any public Harvard sentence list works (search "Harvard sentences IEEE" for the standard set).
2. **Emotional range** (~20-30 min) — re-read a subset of the same sentences in different deliveries:
   neutral, happy, sad, angry, excited. Label which take is which in your own notes (filename or a
   spoken slate at the start of each take) so you can review by ear later if needed.
3. **Conversational / unscripted** (multiple 10-20 min sessions, several days) — just talk. Explain a
   topic you know well, narrate your day, answer imaginary interview questions. This is what teaches
   the model to sound like you *talking*, not you *reading*.
4. **Edge cases** (~10-15 min) — read sentences with numbers, abbreviations, questions, exclamations:
   these are what TTS models most often mispronounce or misprosody.

## Per-session hygiene

- Do a 5-10 second silent "room tone" recording at the start of each session (helps if you ever need
  reference noise for denoising, though the pipeline prefers clean recordings over denoising fixes).
- State the session name/date out loud at the very start ("harvard session one, [date]") — makes it
  trivial to keep raw files straight later.
- Don't record more than ~30-40 minutes in one sitting without a break — vocal fatigue changes your
  voice in ways you won't notice in the moment but will hear in the data.
