import Base
import ./guard.bend as G

# Command-line glue around the proven checker G.check. Trusted, kept small: it only parses the text protocol,
# calls G.check on every paragraph, and prints the result.
#
# Usage: guard <input file>. One paragraph per line, words are numbers chosen by the caller:
#   line   = slots joined by ';', then '|', then the proposal numbers joined by ' '
#   slot   = "<word> <flag 0|1> <cand,cand,...>"   ("-" when there are no candidates)
#   e.g.     1 0 -;2 1 7,8;3 1 9|5 8 4     ->  the output line is  1 8 3
# One output line per input line: the checked word numbers joined by ' '. Malformed input stops the program
# with a message on stderr and a non-zero exit code; nothing is guessed.

type Para is Data:
  Para{slots: List<&2, G.Slot>, props: List<&2, Nat>}

def read_nats(ss: List<&2, String>) -> Maybe<&2, List<&2, Nat>>:
  match ss:
    case Nil{}:
      Some{Nil{}}
    case h <> t:
      do Maybe<&2, List<&2, Nat>>:
        n : Nat <- Nat.read(h)
        r : List<&2, Nat> <- read_nats(t)
        return n <> r

# "" is the empty list, otherwise the numbers are separated by sep.
def nats(s: String, sep: Char) -> Maybe<&2, List<&2, Nat>>:
  match s:
    case SNil{}:
      Some{Nil{}}
    case SCon{h, t}:
      read_nats(String.split(SCon{h, t}, sep))

def cands(s: String) -> Maybe<&2, List<&2, Nat>>:
  match s:
    case SCon{'-', SNil{}}:
      Some{Nil{}}
    case _:
      nats(s, ',')

def flag(s: String) -> Maybe<&2, Bool>:
  match s:
    case SCon{'0', SNil{}}:
      Some{False{}}
    case SCon{'1', SNil{}}:
      Some{True{}}
    case _:
      None{}

def slot3(xs: List<&2, String>) -> Maybe<&2, G.Slot>:
  match xs:
    case w <> (f <> (c <> Nil{})):
      do Maybe<&2, G.Slot>:
        wn : Nat <- Nat.read(w)
        fl : Bool <- flag(f)
        cs : List<&2, Nat> <- cands(c)
        return G.Slot{wn, fl, cs}
    case _:
      None{}

def slot(s: String) -> Maybe<&2, G.Slot>:
  slot3(String.split(s, ' '))

def read_slots(ss: List<&2, String>) -> Maybe<&2, List<&2, G.Slot>>:
  match ss:
    case Nil{}:
      Some{Nil{}}
    case h <> t:
      do Maybe<&2, List<&2, G.Slot>>:
        x : G.Slot <- slot(h)
        r : List<&2, G.Slot> <- read_slots(t)
        return x <> r

def slots(s: String) -> Maybe<&2, List<&2, G.Slot>>:
  match s:
    case SNil{}:
      Some{Nil{}}
    case SCon{h, t}:
      read_slots(String.split(SCon{h, t}, ';'))

def para2(xs: List<&2, String>) -> Maybe<&2, Para>:
  match xs:
    case a <> (b <> Nil{}):
      do Maybe<&2, Para>:
        ss : List<&2, G.Slot> <- slots(a)
        ps : List<&2, Nat> <- nats(b, ' ')
        return Para{ss, ps}
    case _:
      None{}

def para(s: String) -> Maybe<&2, Para>:
  para2(String.split(s, '|'))

# All lines or none. The empty piece after the final newline is not a paragraph.
def paras(ls: List<&2, String>) -> Maybe<&2, List<&2, Para>>:
  match ls:
    case Nil{}:
      Some{Nil{}}
    case SNil{} <> Nil{}:
      Some{Nil{}}
    case h <> t:
      do Maybe<&2, List<&2, Para>>:
        p : Para <- para(h)
        r : List<&2, Para> <- paras(t)
        return p <> r

def show_nats(xs: List<&2, Nat>) -> List<&2, String>:
  match xs:
    case Nil{}:
      Nil{}
    case h <> t:
      Nat.show(h) <> show_nats(t)

def out_line(p: Para) -> String:
  match p:
    case Para{ss, ps}:
      String.join(show_nats(G.check(ss, ps)), " ")

def out_seq(ps: List<&2, Para>) -> List<&2, String>:
  match ps:
    case Nil{}:
      Nil{}
    case h <> t:
      out_line(h) <> out_seq(t)

# Fork-join over the two halves of the paragraph list, d levels deep (2^d tasks), then a plain loop.
def out_par(+d: Nat, +ps: List<&2, Para>) -> List<&2, String>:
  match d:
    case 0n:
      out_seq(ps)
    case 1n+q:
      +h = Nat.div(List.length(&2, Para, ps), 2n)
      a b = out_par(q, List.take(&2, Para, ps, h)) out_par(q, List.drop(&2, Para, ps, h))
      List.append(&2, String, a, b)

def lines_out(ls: List<&2, String>) -> String:
  match ls:
    case Nil{}:
      SNil{}
    case h <> t:
      String.append(h, SCon{'\n', lines_out(t)})

def print_paras(m: Maybe<&2, List<&2, Para>>) -> IO(Unit):
  match m:
    case None{}:
      IO.die(Unit, 1, "guard: malformed input")
    case Some{ps}:
      IO.write(lines_out(out_par(6n, ps)))

def show(text: String) -> IO(Unit):
  print_paras(paras(String.lines(text)))

def show_read(rd: File & Result<&1, &1, U32 & String, String>) -> IO(Unit):
  (f, r) = rd
  do IO<Unit>:
    File.close(f)
    text : String <- IO.pass(String, r)
    show(text)

def read_all(sz: File & Result<&1, &1, U32 & String, U32>) -> IO(Unit):
  (f, r) = sz
  do IO<Unit>:
    n : U32 <- IO.pass(U32, r)
    rd : File & Result<&1, &1, U32 & String, String> <- File.read(f, n)
    show_read(rd)

def run(path: String) -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path, "r"))
    sz : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    read_all(sz)

def with_args(args: List<String>) -> IO(Unit):
  match args:
    case _ <> (path <> _):
      run(path)
    case _:
      IO.die(Unit, 2, "usage: guard <input file>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    with_args(args)
