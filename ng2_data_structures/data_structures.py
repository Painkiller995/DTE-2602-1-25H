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
    def __init__(self):
        """Initialize stack object, with head attribute"""
        pass

    def push(self, data: Any) -> None:
        """Add new node with data to stack"""
        pass

    def peek(self) -> Node | None:
        """Return data from node on top of stack, without changing stack"""
        pass

    def pop(self) -> Node:
        """Remove last added node and return its data"""
        pass


class Queue:
    def __init__(self):
        """Initialize queue object with head and tail"""
        pass

    def enqueue(self, data: Any) -> None:
        """Add node with data to queue"""
        pass

    def peek(self) -> Node | None:
        """Return data from head of queue without changing the queue"""
        pass

    def dequeue(self) -> Node:
        """Remove node from head of queue and return its data"""
        pass


class EmergencyRoomQueue:
    def __init__(self):
        """Initialize emergency room queue, use heapdict as property 'queue'"""
        pass

    def add_patient_with_priority(self, patient_name: str, priority: int) -> None:
        """Add patient name and priority to queue

        # Arguments:
        patient_name:   String with patient name
        priority:       Integer. Higher priority corresponds to lower-value number.
        """
        pass

    def update_patient_priority(self, patient_name: str, new_priority: int) -> None:
        """Update the priority of a patient which is already in the queue

        # Arguments:
        patient_name:   String, name of patient in queue
        new_priority:   Integer, updated priority for patient

        """
        pass

    def get_next_patient(self) -> str:
        """Remove highest-priority patient from queue and return patient name

        # Returns:
        patient_name    String, name of patient with highest priority
        """
        pass


class BinarySearchTree:
    def __init__(self, root: binarytree.Node | None = None):
        """Initialize binary search tree

        # Inputs:
        root:    (optional) An instance of binarytree.Node which is the root of the tree

        # Notes:
        If a root is supplied, validate that the tree meets the requirements
        of a binary search tree (see property binarytree.Node.is_bst ). If not, raise
        ValueError.
        """
        pass

    def insert(self, value: float | int) -> None:
        """Insert a new node into the tree (binarytree.Node object)

        # Inputs:
        value:    Value of new node

        # Notes:
        The method should issue a warning if the value already exists in the tree.
        See https://docs.python.org/3/library/warnings.html#warnings.warn
        In the case of duplicate values, leave the tree unchanged.
        """
        pass

    def __str__(self) -> str | None:
        """Return string representation of tree (helper function for debugging)"""
        if self.root is not None:
            return str(self.root)
