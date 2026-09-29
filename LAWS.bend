import Base
import ./guard.bend as G

# LAWS for the transcript checker (guard.bend). Drafted from the rules the user approved, approved by the user
# on 2026-09-29; the user owns
# this file and the AI does not edit it. PROOF.bend proves every law; `bend PROOF.bend` must print
# ALL PROOFS CHECK before any commit.
#
# A paragraph is a list of slots, one per word, in order: G.Slot{word, flagged, cands} (see guard.bend).
# The AI proposes one word per slot (ps); G.check(ss, ps) returns the words the user is shown.

# --- helpers that only state the laws ---------------------------------------------------------

# Is x one of the words in xs?
def one_of(+x: Nat, xs: List<&2, Nat>) -> Bool:
  List.contains(~Nat, ~Nat.is_eq, xs, x)

# Per slot: the output o keeps a word the speech model was sure about.
def sure_kept(s: G.Slot, +o: Nat) -> Bool:
  match s:
    case G.Slot{w, f, cs}:
      Bool.or(f, Nat.is_eq(o, w))

# Per slot: the output o is the original word, or one of the speech model's own alternatives.
def from_asr(s: G.Slot, +o: Nat) -> Bool:
  match s:
    case G.Slot{w, f, cs}:
      Bool.or(Nat.is_eq(o, w), one_of(o, cs))

# Per slot: may the proposal p replace the word? Only if the word is flagged and p is an alternative.
def may_replace(s: G.Slot, +p: Nat) -> Bool:
  match s:
    case G.Slot{w, f, cs}:
      Bool.and(f, one_of(p, cs))

# If c then yes else no.
def cond(c: Bool, yes: Bool, no: Bool) -> Bool:
  match c:
    case True{}:
      yes
    case False{}:
      no

# Per slot, given the proposal p: a rejected proposal leaves the original word (rule 4), and an allowed
# one is applied (so a checker that never changes anything does not pass).
def decided(+s: G.Slot, +p: Nat, +o: Nat) -> Bool:
  cond(may_replace(s, p), Nat.is_eq(o, p), Nat.is_eq(o, G.Slot.word(s)))

# Whole paragraph: ok holds at every slot, pairing each slot with its output word. False if the lengths
# differ.
def every(~ok: G.Slot -> @+o: Nat -> Bool, ss: List<&2, G.Slot>, out: List<&2, Nat>) -> Bool:
  match ss out:
    case Nil{} Nil{}:
      True{}
    case Con{s, st} Con{o, ot}:
      Bool.and(ok(s, o), every(~ok, st, ot))
    case Nil{} Con{o, ot}:
      False{}
    case Con{s, st} Nil{}:
      False{}

# Whole paragraph, with the proposals: every slot is decided. A missing proposal counts as the
# original word (it can never be allowed to change anything).
def every_decided(ss: List<&2, G.Slot>, ps: List<&2, Nat>, out: List<&2, Nat>) -> Bool:
  match ss ps out:
    case Nil{} qs Nil{}:
      True{}
    case Con{s, st} Nil{} Con{o, ot}:
      Bool.and(Nat.is_eq(o, G.Slot.word(s)), every_decided(st, Nil{}, ot))
    case Con{s, st} Con{p, pt} Con{o, ot}:
      Bool.and(decided(s, p, o), every_decided(st, pt, ot))
    case Nil{} qs Con{o, ot}:
      False{}
    case Con{s, st} qs Nil{}:
      False{}

# --- the laws -------------------------------------------------------------------------------------

# Rule 1. A word the speech model was sure about (not flagged) is never changed.
law sure_words_unchanged:
  for ss: List<&2, G.Slot>
  for ps: List<&2, Nat>
  {every(~sure_kept, ss, G.check(ss, ps)) == True{} : Bool}

# Rule 2. Every word in the result is the original word or one of the speech model's own alternatives
# for that position: the AI never writes free text.
law only_asr_guesses:
  for ss: List<&2, G.Slot>
  for ps: List<&2, Nat>
  {every(~from_asr, ss, G.check(ss, ps)) == True{} : Bool}

# Rule 3. No word is added or removed: the result has exactly one word per slot.
law same_word_count:
  for ss: List<&2, G.Slot>
  for ps: List<&2, Nat>
  {List.length(&2, Nat, G.check(ss, ps)) == List.length(&2, G.Slot, ss) : Nat}

# Rule 4. A rejected suggestion leaves its word exactly as it was, and an allowed one is applied.
law rejected_untouched:
  for ss: List<&2, G.Slot>
  for ps: List<&2, Nat>
  {every_decided(ss, ps, G.check(ss, ps)) == True{} : Bool}
