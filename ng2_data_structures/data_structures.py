"""
This module solves the exercises in assignment 2 of DTE-2602-1-25H.

This implementation could be improved in the feature please check github for the latest version.
https://github.com/Painkiller995/DTE-2602-1-25H

"""

from __future__ import annotations
import warnings
from typing import Any
import binarytree
import heapdict


# Node class (do not change)
class Node:
    """
    Class for a node in a linked list

    Attributes:
    data:   Data stored in node
    next:   Pointer to next node in linked list
    """

    def __init__(self, data: Any = None, next: None | Node = None):
        self.data = data
        self.next = next


# Add your implementations below


class Stack:
    """
    Class for stack data structure (LIFO - last in, first out)
    """

    def __init__(self):
        """Initialize stack object, with head attribute"""
        self.head = None

    def push(self, data: Any) -> None:
        """Add new node with data to stack"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def peek(self) -> Node | None:
        """Return data from node on top of stack, without changing stack"""
        if not self.head:
            raise IndexError("Peek from empty stack")
        return self.head.data

    def pop(self) -> Node | None:
        """Remove last added node and return its data"""
        if not self.head:
            raise IndexError("Pop from empty stack")
        popped_node = self.head
        self.head = self.head.next
        return popped_node.data


class Queue:
    """
    Class for queue data structure (FIFO - first in, first out)
    """

    def __init__(self):
        """Initialize queue object with head and tail"""
        self.head = None
        self.tail = None

    def enqueue(self, data: Any) -> None:
        """Add node with data to queue"""
        new_node = Node(data)
        if self.tail:
            self.tail.next = new_node
        self.tail = new_node
        if not self.head:
            self.head = new_node

    def peek(self) -> Node | None:
        """Return data from head of queue without changing the queue"""
        if not self.head:
            raise IndexError("Peek from empty queue")
        return self.head.data

    def dequeue(self) -> Node | None:
        """Remove node from head of queue and return its data"""
        if not self.head:
            raise IndexError("Dequeue from empty queue")
        dequeued_node = self.head
        self.head = self.head.next
        if not self.head:
            self.tail = None
        return dequeued_node.data


class EmergencyRoomQueue:
    """
    Class for emergency room queue, using heapdict to manage patient priorities
    """

    def __init__(self):
        """Initialize emergency room queue, use heapdict as property 'queue'"""
        self.queue = heapdict.heapdict()

    def add_patient_with_priority(self, patient_name: str, priority: int) -> None:
        """Add patient name and priority to queue

        # Arguments:
        patient_name:   String with patient name
        priority:       Integer. Higher priority corresponds to lower-value number.
        """
        self.queue[patient_name] = priority

    def update_patient_priority(self, patient_name: str, new_priority: int) -> None:
        """Update the priority of a patient which is already in the queue

        # Arguments:
        patient_name:   String, name of patient in queue
        new_priority:   Integer, updated priority for patient

        """
        if patient_name in self.queue:
            self.queue[patient_name] = new_priority
        else:
            warnings.warn(f"Patient {patient_name} not found in queue.")

    def get_next_patient(self) -> str:
        """Remove highest-priority patient from queue and return patient name

        # Returns:
        patient_name    String, name of patient with highest priority
        """
        if self.queue:
            patient_name, _ = self.queue.popitem()
            return patient_name
        raise IndexError("No patients in queue.")


class BinarySearchTree:
    """
    Class for binary search tree, using binarytree library
    """

    def __init__(self, root: binarytree.Node | None = None):
        """Initialize binary search tree

        # Inputs:
        root:    (optional) An instance of binarytree.Node which is the root of the tree

        # Notes:
        If a root is supplied, validate that the tree meets the requirements
        of a binary search tree (see property binarytree.Node.is_bst ). If not, raise
        ValueError.
        """
        if root and not root.is_bst:
            raise ValueError(
                "The supplied node is not the root of a binary search tree."
            )
        self.root = root

    def insert(self, value: float | int) -> None:
        """Insert a new node into the tree (binarytree.Node object)

        # Inputs:
        value:    Value of new node

        # Notes:
        The method should issue a warning if the value already exists in the tree.
        See https://docs.python.org/3/library/warnings.html#warnings.warn
        In the case of duplicate values, leave the tree unchanged.
        """

        if self.root is None:
            self.root = binarytree.Node(value)
            return

        current_node = self.root

        while True:
            if value < current_node.value:
                if not current_node.left:
                    current_node.left = binarytree.Node(value)
                    return
                current_node = current_node.left
            elif value > current_node.value:
                if not current_node.right:
                    current_node.right = binarytree.Node(value)
                    return
                current_node = current_node.right
            else:
                warnings.warn(f"Value {value} already exists in the tree.")
                return

    def __str__(self) -> str:
        """Return string representation of tree (helper function for debugging)"""
        if not self.root:
            return ""
        return str(self.root)
