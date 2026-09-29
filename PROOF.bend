import Base
import ./LAWS.bend as Laws
import ./guard.bend as G

# --- small lemmas -------------------------------------------------------------------------------

# A number equals itself.
def is_eq_refl(x: Nat) -> {Nat.is_eq(x, x) == True{} : Bool}:
  match x:
    case 0n:
      {==}
    case 1n+p:
      is_eq_refl(p)

# `or` with True on the right is True.
def or_true(b: Bool) -> {Bool.or(b, True{}) == True{} : Bool}:
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

# Rule 2, one slot with a flagged word: the picked word is the original or an alternative.
def pick_asr(-cs: List<&2, Nat>, +p: Nat, +w: Nat, b: Bool, e: {List.contains(~Nat, ~Nat.is_eq, cs, p) == b : Bool})
  -> {Bool.or(Nat.is_eq(G.choose(b, p, w), w), List.contains(~Nat, ~Nat.is_eq, cs, G.choose(b, p, w))) == True{} : Bool}:
  match b:
    case True{}:
      %Equal.sym(Bool, List.contains(~Nat, ~Nat.is_eq, cs, p), True{}, e) : {Bool.or(Nat.is_eq(p, w), _) == True{} : Bool}
      or_true(Nat.is_eq(p, w))
    case False{}:
      %Equal.sym(Bool, Nat.is_eq(w, w), True{}, is_eq_refl(w)) : {Bool.or(_, List.contains(~Nat, ~Nat.is_eq, cs, w)) == True{} : Bool}
      {==}

# Rule 4, one slot with a flagged word: an allowed proposal is applied, a rejected one is not.
def pick_decided(+p: Nat, +w: Nat, b: Bool)
  -> {Laws.cond(b, Nat.is_eq(G.choose(b, p, w), p), Nat.is_eq(G.choose(b, p, w), w)) == True{} : Bool}:
  match b:
    case True{}:
      is_eq_refl(p)
    case False{}:
      is_eq_refl(w)

# `and` of two Trues is True.
def and_t(-a: Bool, -b: Bool, ea: {a == True{} : Bool}, eb: {b == True{} : Bool}) -> {Bool.and(a, b) == True{} : Bool}:
  %Equal.sym(Bool, a, True{}, ea) : {Bool.and(_, b) == True{} : Bool}
  eb

# --- the laws -------------------------------------------------------------------------------------

# Rule 1. A slot's own word passes `sure_kept`, flagged or not.
def sure_self(+w: Nat, f: Bool, -cs: List<&2, Nat>) -> {Laws.sure_kept(G.Slot{w, f, cs}, w) == True{} : Bool}:
  match f:
    case True{}:
      {==}
    case False{}:
      is_eq_refl(w)

def sure_pick(+w: Nat, f: Bool, -cs: List<&2, Nat>, +p: Nat)
  -> {Laws.sure_kept(G.Slot{w, f, cs}, G.pick(G.Slot{w, f, cs}, p)) == True{} : Bool}:
  match f:
    case True{}:
      {==}
    case False{}:
      is_eq_refl(w)

def Laws.sure_words_unchanged(ss, ps):
  match ss ps:
    case Nil{} qs:
      {==}
    case Con{G.Slot{w, f, cs}, st} Nil{}:
      and_t(Laws.sure_kept(G.Slot{w, f, cs}, w), Laws.every(~Laws.sure_kept, st, G.check(st, Nil{})),
        sure_self(w, f, cs), Laws.sure_words_unchanged(st, Nil{}))
    case Con{G.Slot{w, f, cs}, st} Con{p, pt}:
      and_t(Laws.sure_kept(G.Slot{w, f, cs}, G.pick(G.Slot{w, f, cs}, p)), Laws.every(~Laws.sure_kept, st, G.check(st, pt)),
        sure_pick(w, f, cs, p), Laws.sure_words_unchanged(st, pt))

# Rule 2. The unchanged word is one of the speech model's guesses.
def asr_self(+w: Nat, f: Bool, -cs: List<&2, Nat>) -> {Laws.from_asr(G.Slot{w, f, cs}, w) == True{} : Bool}:
  %Equal.sym(Bool, Nat.is_eq(w, w), True{}, is_eq_refl(w)) : {Bool.or(_, Laws.one_of(w, cs)) == True{} : Bool}
  {==}

def asr_pick(+w: Nat, f: Bool, +cs: List<&2, Nat>, +p: Nat)
  -> {Laws.from_asr(G.Slot{w, f, cs}, G.pick(G.Slot{w, f, cs}, p)) == True{} : Bool}:
  match f:
    case True{}:
      pick_asr(cs, p, w, List.contains(~Nat, ~Nat.is_eq, cs, p), {==})
    case False{}:
      asr_self(w, False{}, cs)

def Laws.only_asr_guesses(ss, ps):
  match ss ps:
    case Nil{} qs:
      {==}
    case Con{G.Slot{w, f, cs}, st} Nil{}:
      and_t(Laws.from_asr(G.Slot{w, f, cs}, w), Laws.every(~Laws.from_asr, st, G.check(st, Nil{})),
        asr_self(w, f, cs), Laws.only_asr_guesses(st, Nil{}))
    case Con{G.Slot{w, f, cs}, st} Con{p, pt}:
      and_t(Laws.from_asr(G.Slot{w, f, cs}, G.pick(G.Slot{w, f, cs}, p)), Laws.every(~Laws.from_asr, st, G.check(st, pt)),
        asr_pick(w, f, cs, p), Laws.only_asr_guesses(st, pt))

# Rule 3. One word out per slot.
def Laws.same_word_count(ss, ps):
  match ss ps:
    case Nil{} qs:
      {==}
    case Con{s, st} Nil{}:
      %Laws.same_word_count(st, Nil{}) : {1n+List.length(&2, Nat, G.check(st, Nil{})) == 1n+_ : Nat}
      {==}
    case Con{s, st} Con{p, pt}:
      %Laws.same_word_count(st, pt) : {1n+List.length(&2, Nat, G.check(st, pt)) == 1n+_ : Nat}
      {==}

# Rule 4. A rejected proposal keeps the word, an allowed one is applied.
def dec_pick(+w: Nat, f: Bool, +cs: List<&2, Nat>, +p: Nat)
  -> {Laws.decided(G.Slot{w, f, cs}, p, G.pick(G.Slot{w, f, cs}, p)) == True{} : Bool}:
  match f:
    case True{}:
      pick_decided(p, w, List.contains(~Nat, ~Nat.is_eq, cs, p))
    case False{}:
      is_eq_refl(w)

def Laws.rejected_untouched(ss, ps):
  match ss ps:
    case Nil{} qs:
      {==}
    case Con{G.Slot{w, f, cs}, st} Nil{}:
      and_t(Nat.is_eq(w, w), Laws.every_decided(st, Nil{}, G.check(st, Nil{})),
        is_eq_refl(w), Laws.rejected_untouched(st, Nil{}))
    case Con{G.Slot{w, f, cs}, st} Con{p, pt}:
      and_t(Laws.decided(G.Slot{w, f, cs}, p, G.pick(G.Slot{w, f, cs}, p)), Laws.every_decided(st, pt, G.check(st, pt)),
        dec_pick(w, f, cs, p), Laws.rejected_untouched(st, pt))
