import Base

# One word of a transcript paragraph, as the checker sees it. Words are numbers: before calling the
# checker, Python gives each distinct word of the paragraph its own number.
#   word    : the word the speech model wrote
#   flagged : True when the speech model was unsure about this word
#   cands   : the speech model's own alternatives for this word
type Slot is Data:
  Slot{word: Nat, flagged: Bool, cands: List<&2, Nat>}

def Slot.word(s: Slot) -> Nat:
  match s:
    case Slot{w, f, cs}:
      w

# True when p may replace the slot's word: the word is flagged and p is one of its alternatives.
def allowed(s: Slot, +p: Nat) -> Bool:
  match s:
    case Slot{w, f, cs}:
      Bool.and(f, List.contains(~Nat, ~Nat.is_eq, cs, p))

def choose(ok: Bool, p: Nat, w: Nat) -> Nat:
  match ok:
    case True{}:
      p
    case False{}:
      w

def pick(+s: Slot, +p: Nat) -> Nat:
  choose(allowed(s, p), p, Slot.word(s))

# The checked words, slot by slot: the AI's proposal where it is allowed, otherwise the original word.
# A missing proposal keeps the original word; extra proposals are ignored.
def check(ss: List<&2, Slot>, ps: List<&2, Nat>) -> List<&2, Nat>:
  match ss ps:
    case Nil{} qs:
      Nil{}
    case Con{s, st} Nil{}:
      Slot.word(s) <> check(st, Nil{})
    case Con{s, st} Con{p, pt}:
      pick(s, p) <> check(st, pt)
