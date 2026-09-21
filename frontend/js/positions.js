// Display names and one-line explanations per base position.
// "who" describes which athlete the trailing 1/2 refers to, following the dataset authors'
// wording where they document it.

export const POSITIONS = [
  { base: "standing", name: "Standing", who: null,
    text: "Both athletes on their feet, before the fight hits the mat." },
  { base: "takedown", name: "Takedown", who: "athlete taking the other down",
    text: "The transition from standing to the ground." },
  { base: "open_guard", name: "Open guard", who: "athlete in the guard position",
    text: "Bottom athlete controls with legs and grips, ankles not locked." },
  { base: "closed_guard", name: "Closed guard", who: "athlete in the guard position",
    text: "Bottom athlete locks the legs around the opponent's waist." },
  { base: "half_guard", name: "Half guard", who: "athlete in the guard position",
    text: "Bottom athlete traps one of the opponent's legs between their own." },
  { base: "5050_guard", name: "50/50 guard", who: null,
    text: "Both athletes entangle the same leg — mirror image, so no side is on top." },
  { base: "side_control", name: "Side control", who: "athlete on top",
    text: "Top athlete has passed the legs and pins across the chest." },
  { base: "mount", name: "Mount", who: "athlete on top",
    text: "Top athlete sits on the torso, knees on the mat." },
  { base: "back", name: "Back", who: "athlete controlling the back",
    text: "One athlete behind the other, usually with hooks in — the most dominant position." },
  { base: "turtle", name: "Turtle", who: "athlete on top",
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
