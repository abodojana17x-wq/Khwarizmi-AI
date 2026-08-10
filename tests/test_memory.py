"""Tests for RAFIQ Phase 05 — Memory System."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from rafig.memory import (
    ConversationMemory,
    EpisodicMemory,
    Importance,
    MemoryDiagnostics,
    MemoryEntry,
    MemoryManager,
    ProjectMemory,
    SemanticMemory,
    WorkingMemory,
)


class WorkingMemoryTests(unittest.TestCase):
    def test_store_and_retrieve(self) -> None:
        wm = WorkingMemory(capacity=5)
        entry = wm.store("Hello world", source="test")
        self.assertEqual(wm.size(), 1)
        self.assertEqual(entry.content, "Hello world")
        self.assertEqual(entry.source, "test")

    def test_capacity_eviction(self) -> None:
        wm = WorkingMemory(capacity=3)
        first = wm.store("first")
        wm.store("second")
        wm.store("third")
        wm.store("fourth")
        self.assertEqual(wm.size(), 3)
        self.assertIsNone(wm.get(first.id))

    def test_update_entry(self) -> None:
        wm = WorkingMemory()
        entry = wm.store("old content")
        updated = wm.update(entry.id, content="new content")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "new content")

    def test_delete_entry(self) -> None:
        wm = WorkingMemory()
        entry = wm.store("delete me")
        self.assertTrue(wm.delete(entry.id))
        self.assertFalse(wm.delete(entry.id))
        self.assertEqual(wm.size(), 0)

    def test_clear_returns_count(self) -> None:
        wm = WorkingMemory()
        wm.store("a")
        wm.store("b")
        wm.store("c")
        self.assertEqual(wm.clear(), 3)
        self.assertEqual(wm.size(), 0)

    def test_retrieve_by_query(self) -> None:
        wm = WorkingMemory()
        wm.store("Python is great", tags=["python"])
        wm.store("JavaScript is fine", tags=["js"])
        results = wm.retrieve("Python")
        self.assertTrue(results)
        self.assertIn("Python", results[0].content)


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "conv.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_turn_and_retrieve(self) -> None:
        conv = ConversationMemory(self.db_path, max_turns=10)
        try:
            conv.add_turn("user", "Hi there!")
            conv.add_turn("assistant", "Hello! How can I help?")
            self.assertEqual(conv.size(), 2)
            recent = conv.recent(10)
            self.assertEqual(len(recent), 2)
            self.assertEqual(recent[0].content, "Hi there!")
            self.assertEqual(recent[1].content, "Hello! How can I help?")
        finally:
            conv.close()

    def test_max_turns_eviction(self) -> None:
        conv = ConversationMemory(self.db_path, max_turns=3)
        try:
            for i in range(5):
                conv.add_turn("user", f"turn {i}")
            self.assertEqual(conv.size(), 3)
        finally:
            conv.close()

    def test_search_by_query(self) -> None:
        conv = ConversationMemory(self.db_path)
        try:
            conv.add_turn("user", "Fix this Python code")
            conv.add_turn("user", "Tell me a joke")
            results = conv.retrieve("Python")
            self.assertTrue(results)
            self.assertIn("Python", results[0].content)
        finally:
            conv.close()

    def test_update_turn(self) -> None:
        conv = ConversationMemory(self.db_path)
        try:
            entry = conv.add_turn("user", "original")
            updated = conv.update(entry.id, content="modified")
            self.assertIsNotNone(updated)
            self.assertEqual(updated.content, "modified")
        finally:
            conv.close()

    def test_delete_turn(self) -> None:
        conv = ConversationMemory(self.db_path)
        try:
            entry = conv.add_turn("user", "delete me")
            self.assertTrue(conv.delete(entry.id))
            self.assertEqual(conv.size(), 0)
        finally:
            conv.close()


class PersistentMemoryTests(unittest.TestCase):
    """Tests shared by EpisodicMemory, SemanticMemory, ProjectMemory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "store.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_store(self, cls: type) -> "SQLiteMemory":  # type: ignore[name-defined]
        return cls(self.db_path, max_entries=100)

    def test_episodic_store_and_retrieve(self) -> None:
        store = self._make_store(EpisodicMemory)
        try:
            entry = store.store("User asked about Python loops", context="session 1")
            self.assertEqual(store.size(), 1)
            results = store.retrieve("Python")
            self.assertTrue(results)
            self.assertEqual(results[0].content, entry.content)
        finally:
            store.close()

    def test_semantic_store_and_search(self) -> None:
        store = self._make_store(SemanticMemory)
        try:
            store.store("Python uses indentation for blocks", importance=Importance.HIGH)
            store.store("JavaScript uses curly braces", importance=Importance.NORMAL)
            results = store.retrieve("Python")
            self.assertTrue(results)
            self.assertIn("Python", results[0].content)
        finally:
            store.close()

    def test_project_store(self) -> None:
        store = self._make_store(ProjectMemory)
        try:
            store.store("rafig/memory/ holds the Phase 05 code", tags=["structure"])
            self.assertEqual(store.size(), 1)
        finally:
            store.close()

    def test_max_entries_eviction(self) -> None:
        store = EpisodicMemory(self.db_path, max_entries=3)
        try:
            for i in range(5):
                store.store(f"event {i}")
            self.assertEqual(store.size(), 3)
        finally:
            store.close()

    def test_update_and_delete(self) -> None:
        store = self._make_store(SemanticMemory)
        try:
            entry = store.store("fact A")
            updated = store.update(entry.id, content="fact A updated")
            self.assertIsNotNone(updated)
            self.assertEqual(updated.content, "fact A updated")
            self.assertTrue(store.delete(entry.id))
            self.assertEqual(store.size(), 0)
        finally:
            store.close()


class MemoryEntryTests(unittest.TestCase):
    def test_relevance_score_with_terms(self) -> None:
        entry = MemoryEntry(
            content="Python loops are fun",
            tags=["python"],
            importance=Importance.HIGH,
        )
        score = entry.relevance_score(["Python", "loops"])
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_relevance_score_no_terms(self) -> None:
        entry = MemoryEntry(content="anything")
        self.assertEqual(entry.relevance_score([]), 0.5)

    def test_touch_increments_access(self) -> None:
        entry = MemoryEntry(content="x")
        entry.touch()
        self.assertEqual(entry.access_count, 1)


class MemoryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "manager.db"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manager_stores_and_searches(self) -> None:
        with MemoryManager(db_path=self.db_path) as mgr:
            mgr.conversation.add_turn("user", "How do I write a for loop in Python?")
            mgr.semantic.store("Python for loops iterate over iterables")
            mgr.episodic.store("User struggled with list comprehensions")
            hits = mgr.search("Python")
            self.assertTrue(hits)
            stores_seen = {hit.store for hit in hits}
            self.assertIn("conversation", stores_seen)

    def test_manager_clear_all(self) -> None:
        with MemoryManager(db_path=self.db_path) as mgr:
            mgr.working.store("scratch")
            mgr.conversation.add_turn("user", "hi")
            mgr.semantic.store("fact")
            cleared = mgr.clear_all()
            self.assertEqual(cleared["working"], 1)
            self.assertEqual(cleared["conversation"], 1)
            self.assertEqual(cleared["semantic"], 1)
            self.assertEqual(mgr.working.size(), 0)

    def test_manager_diagnostics(self) -> None:
        with MemoryManager(db_path=self.db_path) as mgr:
            mgr.conversation.add_turn("user", "hello")
            diag = mgr.diagnostics()
            self.assertIsInstance(diag, MemoryDiagnostics)
            self.assertEqual(diag.conversation_size, 1)
            self.assertGreater(diag.db_size_bytes, 0)
            text = str(diag)
            self.assertIn("Conversation store", text)


class MemoryIntegrationTests(unittest.TestCase):
    """Integration test: make sure Rafiq itself boots with memory."""

    def test_rafig_starts_with_memory(self) -> None:
        from rafig.config import Settings
        from rafig.rafig import Rafiq

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(project_root=Path(tmp))
            engine = Rafiq(settings=settings)
            engine.start()
            try:
                self.assertIsNotNone(engine.memory)
                engine.memory.conversation.add_turn("user", "integration test")
                engine.memory.semantic.store("rafiq works")
                hits = engine.memory.search("rafiq")
                self.assertTrue(hits)
                engine.run()
            finally:
                engine.shutdown()


if __name__ == "__main__":
    unittest.main()
