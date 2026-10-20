"""Offline tests: a local test website + a scripted stand-in for the LLM.

They check the plumbing (browser, extractor, loop, backtracking, verification) without needing
a real model or internet. Run with:  python -m unittest discover tests
"""

import functools
import re
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from webagent import Browser, WebAgent
from webagent.agent import is_non_answer, is_supported
from webagent.extractor import normalize_url, parse_html, rank_links, top_passages, trim_text
from webagent.llm import parse_json

SITE_DIR = Path(__file__).parent / "site"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class ScriptedLLM:
    """Pretends to be the model: answers when a regex matches the page text, otherwise clicks links
    whose text contains the next keyword. Records every prompt so tests can inspect them."""

    def __init__(self, answer_regex, keywords, bad_evidence_first=False, give_up_on=None, insist_give_up=False,
                 invent_number=None, non_answer_on=None):
        self.answer_regex = re.compile(answer_regex, re.I)
        self.keywords = [k.lower() for k in keywords]
        self.bad_evidence_first = bad_evidence_first
        self.give_up_on = give_up_on  # URL fragment of a page where it wrongly gives up on the whole site
        self.insist_give_up = insist_give_up
        self.give_up_rechecks = 0
        self.invent_number = invent_number  # "first" or "always": real quote + a number that isn't in it
        self.evidence_rechecks = 0
        self.non_answer_on = non_answer_on  # URL fragment of a page where it "answers" None with a real quote
        self.prompts = []

    def decide(self, messages):
        prompt = messages[1]["content"]
        self.prompts.append(prompt)
        if messages[-1]["role"] == "user" and "ends the WHOLE search" in messages[-1]["content"]:
            self.give_up_rechecks += 1
            if self.insist_give_up:
                return {"action": "give_up", "reason": "still sure"}
            return {"action": "back", "reason": "only this page lacks it"}
        page_url = re.search(r"^CURRENT PAGE URL: (\S+)", prompt, re.M).group(1)
        if self.non_answer_on and self.non_answer_on in page_url:
            first_line = prompt.split("PAGE TEXT:\n", 1)[1].splitlines()[0]
            return {"action": "answer", "answer": "None", "evidence": first_line}
        if self.give_up_on and self.give_up_on in page_url:
            return {"action": "give_up", "reason": "The page does not mention it."}
        if messages[-1]["role"] == "user" and "evidence was not found" in messages[-1]["content"]:
            self.evidence_rechecks += 1
            self.bad_evidence_first = False
            if self.invent_number == "first":
                self.invent_number = None
            prompt = messages[1]["content"]
        text = prompt.split("PAGE TEXT:\n", 1)[1].split("\n\nLINKS:\n", 1)[0]
        links = re.findall(r"^\[(\d+)\] (.+?) \((\S+)\)$", prompt.split("\n\nLINKS:\n", 1)[1], re.M)

        match = self.answer_regex.search(text)
        if match:
            evidence = "something not on the page" if self.bad_evidence_first else match.group(0)
            answer = "500" if self.invent_number else match.group(1)
            return {"action": "answer", "answer": answer, "evidence": evidence}
        for keyword in self.keywords:
            for link_id, link_text, _ in links:
                if keyword in link_text.lower():
                    return {"action": "visit", "link_id": int(link_id), "reason": f"contains '{keyword}'"}
        return {"action": "back", "reason": "dead end"}


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handler = functools.partial(QuietHandler, directory=str(SITE_DIR))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.browser = Browser(timeout_ms=10000).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.browser.__exit__(None, None, None)
        cls.server.shutdown()
        cls.server.server_close()

    def agent(self, llm, max_steps=8):
        return WebAgent(llm=llm, browser=self.browser, max_steps=max_steps, verbose=False)

    def test_finds_answer_two_levels_deep(self):
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech admissions", "admissions"])
        result = self.agent(llm).run(self.base + "index.html", "What is the MTech application deadline?")
        self.assertTrue(result.found)
        self.assertEqual(result.answer, "15 December 2026")
        self.assertTrue(result.source.endswith("admissions/mtech.html"))
        self.assertTrue(result.verified)
        self.assertEqual(result.pages_opened, 3)

    def test_reads_javascript_rendered_content(self):
        llm = ScriptedLLM(r"held on (\d+ \w+ \d{4})", ["news"])
        result = self.agent(llm).run(self.base + "index.html", "When is convocation?")
        self.assertEqual(result.answer, "3 March 2027")

    def test_reads_tables(self):
        llm = ScriptedLLM(r"Admissions Office \| ([+\d ]+)", ["contact"])
        result = self.agent(llm).run(self.base, "Phone number of the admissions office?")
        self.assertEqual(result.answer.strip(), "+91 120 555 0199")

    def test_backtracks_from_dead_end_and_never_repeats_pages(self):
        # Keyword sends it to programs/mtech.html first (no deadline there), then it must backtrack.
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech in", "programs"])
        result = self.agent(llm).run(self.base + "index.html", "MTech admissions last date to apply?")
        self.assertTrue(result.found, result.steps)
        self.assertEqual(result.answer, "15 December 2026")
        keys = [normalize_url(u) for u in result.visited]
        self.assertEqual(len(keys), len(set(keys)), "a page was visited twice")

    def test_stops_at_step_limit_when_answer_missing(self):
        llm = ScriptedLLM(r"CEO email: (\S+)", [])
        result = self.agent(llm, max_steps=4).run(self.base + "index.html", "What is the CEO's email?")
        self.assertFalse(result.found)
        self.assertLessEqual(result.pages_opened, 4)

    def test_stays_on_same_website(self):
        llm = ScriptedLLM(r"never matches (x)", ["partner"])
        result = self.agent(llm, max_steps=3).run(self.base + "index.html", "Who is the partner?")
        self.assertFalse(any("example.org" in u for u in result.visited))
        self.assertNotIn("example.org", llm.prompts[0])

    def test_evidence_check_asks_model_to_recheck(self):
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech admissions", "admissions"], bad_evidence_first=True)
        result = self.agent(llm).run(self.base + "index.html", "MTech deadline?")
        self.assertTrue(result.found)
        self.assertTrue(result.verified)

    def test_give_up_on_an_off_topic_page_is_rechecked(self):
        # Real failure (docs.python.org): on an off-topic page the model said "the page does not mention it"
        # and chose give_up, ending the run after 3 of 8 pages. Backtracking would have found the answer.
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["news", "mtech admissions"], give_up_on="news.html")
        result = self.agent(llm).run(self.base + "index.html", "MTech admissions last date to apply?")
        self.assertEqual(llm.give_up_rechecks, 1)
        self.assertTrue(result.found, result.steps)
        self.assertEqual(result.answer, "15 December 2026")

    def test_confirmed_give_up_is_accepted(self):
        llm = ScriptedLLM(r"CEO email: (\S+)", [], give_up_on="index.html", insist_give_up=True)
        result = self.agent(llm).run(self.base + "index.html", "What is the CEO's email?")
        self.assertEqual(llm.give_up_rechecks, 1)
        self.assertFalse(result.found)
        self.assertEqual(result.pages_opened, 1)

    def test_made_up_number_is_rechecked_and_corrected(self):
        # Real failure (sqlite.org): a real quote from the page, paired with a number that isn't in it.
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech admissions", "admissions"], invent_number="first")
        result = self.agent(llm).run(self.base + "index.html", "MTech deadline?")
        self.assertEqual(llm.evidence_rechecks, 1)
        self.assertEqual(result.answer, "15 December 2026")
        self.assertTrue(result.verified)

    def test_unsupported_answer_is_rejected_not_returned(self):
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["mtech admissions", "admissions"], invent_number="always")
        result = self.agent(llm, max_steps=4).run(self.base + "index.html", "MTech deadline?")
        self.assertFalse(result.found, result.steps)
        self.assertIsNone(result.answer)
        self.assertIn("rejected_answer", [s["action"] for s in result.steps])

    def test_none_answer_means_keep_searching(self):
        # Real failure (scrapethissite.com): asked about a country not in the list, the model answered "None",
        # quoting "Capital: None" from another country's row.
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["admissions", "mtech admissions"],
                          non_answer_on="admissions.html")
        result = self.agent(llm).run(self.base + "index.html", "MTech deadline?")
        self.assertEqual(result.answer, "15 December 2026", result.steps)
        self.assertIn("i.e. not found on this page", " ".join(s["detail"] for s in result.steps))

    def test_broken_pages_do_not_use_up_the_page_budget(self):
        # Real failure (ox.ac.uk): 6 of 8 page slots went to HTTP 403s, leaving no room to find the answer.
        llm = ScriptedLLM(r"Last date to apply: ([^\n]+)", ["broken", "mtech admissions", "admissions"])
        result = self.agent(llm, max_steps=3).run(self.base + "links.html", "MTech deadline?")
        self.assertEqual(result.answer, "15 December 2026", result.steps)
        self.assertEqual(result.pages_opened, 3)
        self.assertEqual([s["action"] for s in result.steps].count("error"), 4)

    def test_stops_when_too_many_pages_fail(self):
        llm = ScriptedLLM(r"never (x)", ["broken"])
        agent = WebAgent(llm=llm, browser=self.browser, max_steps=8, verbose=False, max_errors=2)
        result = agent.run(self.base + "links.html", "anything")
        self.assertFalse(result.found)
        self.assertIn("could not be loaded", result.error)

    def test_redirect_to_another_website_is_not_read(self):
        # Real failure (nps.gov): a link redirected to a Microsoft login page, which the agent then read.
        llm = ScriptedLLM(r"Admissions Office \| ([+\d ]+)", [])
        result = self.agent(llm, max_steps=2).run(self.base + "redirect.html", "Phone number of the admissions office?")
        self.assertFalse(result.found)
        self.assertEqual(result.steps[0]["action"], "error")
        self.assertIn("Redirected off the website", result.steps[0]["detail"])

    def test_missing_page_is_reported_not_crashing(self):
        llm = ScriptedLLM(r"never (x)", [])
        result = self.agent(llm, max_steps=2).run(self.base + "does-not-exist.html", "anything")
        self.assertEqual(result.steps[0]["action"], "error")
        self.assertFalse(result.found)


class VerificationTests(unittest.TestCase):
    PAGE = "Maximum Number Of Columns\nThe default setting for SQLITE_MAX_COLUMN is 2,000. You can raise it to 32767."

    def test_numbers_in_answer_must_be_in_the_quote(self):
        quote = "The default setting for SQLITE_MAX_COLUMN is 2,000."
        self.assertTrue(is_supported(quote, "2000", self.PAGE, [self.PAGE]))
        self.assertTrue(is_supported(quote, "2,000 columns", self.PAGE, [self.PAGE]))
        self.assertFalse(is_supported(quote, "500", self.PAGE, [self.PAGE]))
        self.assertTrue(is_supported("Maximum Number Of Columns", "32767", self.PAGE, [self.PAGE]))  # answer is on the page
        self.assertTrue(is_supported(quote, "the column limit", self.PAGE, [self.PAGE]))  # no numbers: quote is enough


    def test_non_answers(self):
        for text in ["None", "none.", "N/A", "Not mentioned", "not found", "No information available", ""]:
            self.assertTrue(is_non_answer(text), text)
        for text in ["Nonesuch Press", "1969", "Not before 9am", "Santa Clara", "No. 10 Downing Street"]:
            self.assertFalse(is_non_answer(text), text)


class ExtractorTests(unittest.TestCase):
    HTML = """<html><head><title>T</title><style>.x{}</style></head><body>
    <nav><a href="/a">Alpha</a></nav><script>var hidden=1;</script>
    <p>Hello   world</p><a href="/a#section">Alpha again</a><a href="mailto:x@y.z">mail</a>
    <a href="/file.pdf">PDF</a><a href="https://other.site/page">Fees and payments</a>
    <table><tr><td>Fee</td><td>100</td></tr></table></body></html>"""

    def test_parse_html(self):
        page = parse_html(self.HTML, "https://site.test/start")
        self.assertIn("Hello world", page.text)
        self.assertIn("Fee | 100", page.text)
        self.assertNotIn("hidden", page.text)
        self.assertTrue(page.text.startswith("Hello world"))  # nav removed from text...
        urls = [l.url for l in page.links]
        self.assertEqual(urls, ["https://site.test/a", "https://other.site/page"])  # ...but kept as link; deduped
        self.assertEqual([l.id for l in page.links], [1, 2])

    def test_adjacent_links_do_not_merge(self):
        page = parse_html('<body><div><a href="/a">Home</a><a href="/b">About</a></div></body>', "https://s.test/")
        self.assertIn("Home About", page.text)

    def test_link_labels_are_readable(self):
        # Real shapes: quotes.toscrape "(about)" links, books.toscrape image link before the title link,
        # webscraper.io pagination with bare numbers and a "›" arrow.
        html = """<body>
        <div class="quote"><span>“A day without sunshine…”</span>
          <span>by <small class="author">Steve Martin</small> <a href="/author/Steve-Martin">(about)</a></span></div>
        <div class="quote"><span>“Try not to become a man of success…”</span>
          <span>by <small class="author">Albert Einstein</small> <a href="/author/Albert-Einstein">(about)</a></span></div>
        <article><a href="olio_984/index.html"><img alt="Olio" src="x.jpg"></a>
          <h3><a href="olio_984/index.html" title="Olio">Olio</a></h3></article>
        <ul class="pagination"><li><a href="?page=2">2</a></li><li><a href="?page=3">3</a></li>
          <li><a href="?page=2" rel="next" aria-label="Next »">›</a></li></ul>
        <p>Founded in <a href="/1969">1969</a>.</p></body>"""
        texts = [l.text for l in parse_html(html, "https://shop.test/cat/").links]
        self.assertIn("(about) [by Steve Martin]", texts)
        self.assertIn("(about) [by Albert Einstein]", texts)
        self.assertIn("Olio", texts)
        self.assertNotIn("index.html", texts)
        self.assertIn("Page 2", texts)
        self.assertIn("Page 3", texts)
        self.assertIn("1969", texts)  # a number outside pagination stays as it is

    def test_rank_links(self):
        page = parse_html(self.HTML, "https://site.test/start")
        self.assertEqual(rank_links(page.links, "what are the fees", 5)[0].text, "Fees and payments")

    def test_site_name_words_do_not_bury_the_right_link(self):
        # Shaped like nps.gov: the question repeats the site's name ("Yellowstone National Park"), dozens of
        # links contain those words, and "Fees & Passes" (plural) comes late in the menu.
        menu = "".join(f'<a href="/yell/learn/yellowstone-topic-{i}.htm">Yellowstone National Park topic {i}</a>'
                       for i in range(60))
        html = f"""<html><head><title>Yellowstone National Park (U.S. National Park Service)</title></head>
        <body>{menu}<a href="/yell/learn/yellowstoneestablishment.htm">Birth of a National Park</a>
        <a href="/yell/planyourvisit/fees.htm">Fees &amp; Passes</a></body></html>"""
        page = parse_html(html, "https://www.nps.test/yell/index.htm")
        question = "How much does a standard entrance pass to Yellowstone National Park cost?"
        self.assertEqual(rank_links(page.links, question, 40, page.title)[0].text, "Fees & Passes")

    def test_trim_keeps_relevant_part(self):
        filler = "\n".join(f"Unrelated line number {i} about nothing in particular." for i in range(400))
        text = "Page heading\n" + filler + "\nThe hostel fee is 60000 per year.\n" + filler
        trimmed = trim_text(text, "hostel fee", 2000)
        self.assertLessEqual(len(trimmed), 2000)
        self.assertIn("hostel fee is 60000", trimmed)
        self.assertTrue(trimmed.startswith("Page heading"))

    def test_trim_prefers_rare_query_words(self):
        # A long docs page where every paragraph mentions "str", "method" and string-ish words;
        # only one paragraph mentions the rare word "removeprefix". It must survive trimming.
        names = ["strip", "lstrip", "rstrip", "split", "rsplit", "startswith", "endswith", "splitlines"]
        blocks = []
        for i in range(250):
            name = names[i % len(names)]
            blocks.append(
                f"str.{name}(chars) This str method returns a copy of the string with characters handled. "
                f"The method works on strings and substrings; stripping uses strict string rules. "
                f"See also the str methods for bytes, the string module and the method resolution order. "
                f"Example {i}: calling the method on a str object returns a new string; the original string "
                f"is unchanged because strings are immutable and every str method returns a new str."
            )
            if i == 180:
                blocks.append(
                    "str.removeprefix(prefix, /) If the string starts with the prefix string, return "
                    "string[len(prefix):]. Otherwise, return a copy of the original string. Added in version 3.9."
                )
        text = "Built-in Types\n" + "\n".join(blocks)
        trimmed = trim_text(text, "What does the str.removeprefix method do?", 6000)
        self.assertIn("str.removeprefix(prefix, /)", trimmed)
        self.assertLessEqual(len(trimmed), 6000)
        highlights = top_passages(text, "What does the str.removeprefix method do?")
        self.assertIn("str.removeprefix(prefix, /)", highlights)
        self.assertLessEqual(len(highlights), 1600)

    def test_docs_style_html_reads_cleanly_and_highlights_the_definition(self):
        # Shaped like docs.python.org: every code token in its own <span>, methods as <dt>/<dd>,
        # and another method's text that mentions str.removeprefix() in passing (the earlier failure).
        def method(name, sig, body):
            spans = "".join(f'<span class="n">{t}</span>' for t in ["str", ".", name, "(", sig, ")"])
            # Real pages have source line breaks inside tags; a browser shows them as spaces.
            return (f'<dl class="py method">\n<dt class="sig sig-object py" id="str.{name}">\n{spans}'
                    f'<a class="headerlink" href="#str.{name}">\u00b6</a>\n</dt>\n<dd><p>{body}\n</p>\n</dd></dl>')
        filler = "".join(
            method(n, "chars", f"Return a copy of the string with characters handled by this str method {i}. "
                                "The method works on strings; see the string module.")
            for i, n in enumerate(["split", "rsplit", "strip", "rstrip", "title", "upper", "zfill"] * 12)
        )
        lstrip = method("lstrip", "chars", 'Return a copy of the string with leading characters removed. '
                        'See <a href="#str.removeprefix"><code>str.removeprefix()</code></a> for a method that will '
                        'remove a single prefix string. <pre><span>&#39;Arthur: three!&#39;</span><span>.</span>'
                        '<span>removeprefix</span><span>(</span><span>&#39;Arthur: &#39;</span><span>)</span></pre>')
        definition = method("removeprefix", "prefix, /", "If the string starts with the prefix string, return "
                            "string[len(prefix):]. Otherwise, return a copy of the original string.")
        html = f"<html><body><h1>Built-in Types</h1>{lstrip}{filler}{definition}{filler}</body></html>"
        page = parse_html(html, "https://docs.test/stdtypes.html")
        self.assertIn("## str.removeprefix(prefix, /)", page.text)          # one readable line, marked as a term
        self.assertIn("'Arthur: three!'.removeprefix('Arthur: ')", page.text)  # code not split into fragments
        self.assertNotIn("\u00b6", page.text)
        first = top_passages(page.text, "What does the str.removeprefix method do?").split("\n---\n")[0]
        self.assertIn("If the string starts with the prefix string", first)

    def test_layout_table_page_keeps_its_lines(self):
        # Shaped like nps.gov: the whole page sits inside a layout <table>, with headings and lists inside
        # a cell, plus a real data table. Flattening the layout row put the page on one line and the
        # fee never reached the model.
        filler = "".join(f"<p>Visitor tip {i}: check road status and weather before you travel.</p>" for i in range(150))
        html = f"""<html><body><table class="CS_Layout"><tr class="CS_Layout_TR">
        <td class="CS_Layout_TD">menu</td><td class="CS_Layout_TD"><section><h2>Fees and Passes</h2>{filler}
        <ul><li>The fee for a Standard entrance pass is $20.00\u2013$35.00.</li></ul>
        <h3>Annual Pass</h3><p>$70.00</p>{filler}
        <table><tr><th>Pass</th><th>Price</th></tr><tr><td>Motorcycle</td><td>$30.00</td></tr></table>
        </section></td></tr></table></body></html>"""
        page = parse_html(html, "https://park.test/fees.htm")
        self.assertLess(max(len(line) for line in page.text.splitlines()), 300)
        self.assertIn("\nThe fee for a Standard entrance pass is $20.00\u2013$35.00.\n", page.text)
        self.assertIn("## Annual Pass", page.text)
        self.assertIn("Motorcycle | $30.00", page.text)  # data tables are still flattened
        question = "How much is a standard entrance pass?"
        self.assertIn("Standard entrance pass is $20.00", trim_text(page.text, question, 6000))
        self.assertIn("Standard entrance pass is $20.00", top_passages(page.text, question))

    def test_trim_handles_one_huge_line(self):
        filler = " ".join(f"Unrelated sentence number {i} about nothing." for i in range(600))
        text = "Heading\n" + filler + " The hostel fee is 60000 per year. " + filler
        self.assertIn("hostel fee is 60000", trim_text(text, "hostel fee", 3000))

    def test_normalize_url(self):
        self.assertEqual(normalize_url("https://www.a.com/x/index.html#y"), normalize_url("http://a.com/x/"))


class ParseJsonTests(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(parse_json('{"action": "back"}')["action"], "back")
        self.assertEqual(parse_json('```json\n{"action": "visit", "link_id": 3}\n```')["link_id"], 3)
        self.assertEqual(parse_json('<think>hmm {"no"}</think> Sure: {"action": "give_up"}')["action"], "give_up")
        with self.assertRaises(ValueError):
            parse_json("no json here")


if __name__ == "__main__":
    unittest.main()
