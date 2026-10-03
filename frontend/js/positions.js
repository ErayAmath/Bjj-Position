// Display names and one-line explanations per base position.
//
// What the trailing 1/2 means was VERIFIED against the dataset, not taken from the paper:
// for every class we compared the hip height of both athletes (y grows downwards).
//   mount1        athlete 1 higher in 100.0 % of frames  -> the number is the TOP athlete
//   side_control1 97.9 %                                 -> top athlete
//   turtle1       75.7 %                                 -> top athlete
//   back1         73.5 %                                 -> the athlete controlling the back
//   closed_guard1  0.0 %, half_guard1 0.1 %, open_guard1 0.1 %
//                                                        -> the number is the athlete PLAYING
//                                                           guard, i.e. the one UNDERNEATH
//   takedown1     17.9 % — not conclusive; the dataset does not document it, so the 1/2 of a
//                 takedown should not be trusted.

export const POSITIONS = [
  { base: "standing", name: "Standing", who: null, suffix: null,
    text: "Both athletes on their feet, before the fight hits the mat." },
  { base: "takedown", name: "Takedown", who: "athlete the dataset attributes it to",
    suffix: "meaning undocumented — do not rely on it",
    text: "The transition from standing to the ground." },
  { base: "open_guard", name: "Open guard", who: "athlete playing guard (underneath)",
    suffix: "who plays guard (bottom)",
    text: "Bottom athlete controls with legs and grips, ankles not locked." },
  { base: "closed_guard", name: "Closed guard", who: "athlete playing guard (underneath)",
    suffix: "who plays guard (bottom)",
    text: "Bottom athlete locks the legs around the opponent's waist." },
  { base: "half_guard", name: "Half guard", who: "athlete playing guard (underneath)",
    suffix: "who plays guard (bottom)",
    text: "Bottom athlete traps one of the opponent's legs between their own." },
  { base: "5050_guard", name: "50/50 guard", who: null, suffix: null,
    text: "Both athletes entangle the same leg — mirror image, so no side is on top." },
  { base: "side_control", name: "Side control", who: "athlete on top", suffix: "who is on top",
    text: "Top athlete has passed the legs and pins across the chest." },
  { base: "mount", name: "Mount", who: "athlete on top", suffix: "who is on top",
    text: "Top athlete sits on the torso, knees on the mat." },
  { base: "back", name: "Back", who: "athlete controlling the back",
    suffix: "who controls the back",
    text: "One athlete behind the other, usually with hooks in — the most dominant position." },
  { base: "turtle", name: "Turtle", who: "athlete on top", suffix: "who is on top",
    text: "Bottom athlete on knees and elbows, protecting the neck." },
];

export function describe(className) {
  const m = className.match(/^(.*?)([12])?$/);
  const base = m[1];
  const athlete = m[2] ? Number(m[2]) : null;
  const pos = POSITIONS.find((p) => p.base === base);
  if (!pos) return { title: className, text: "", athlete };
  const suffix = athlete && pos.who ? ` Label ${athlete}: athlete ${athlete} is the ${pos.who}.` : "";
  return { title: pos.name, text: pos.text + suffix, athlete };
}
