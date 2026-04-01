class BinarySearchableSet:
    def __init__(self):
        # arrays[i] is either [] or a sorted array of length 2^i
        self.arrays = []
        self.n = 0

    def search(self, x):
        """
        Return True if x exists, otherwise False.
        Worst-case: O((log n)^2)
        """
        for arr in self.arrays:
            if arr and self._binary_search(arr, x):
                return True
        return False

    def insert(self, x):
        """
        Insert x into the structure.
        Worst-case: O(n)
        Amortized: O(log n)
        """
        carry = [x]
        i = 0

        while True:
            # extend arrays list if needed
            if i >= len(self.arrays):
                self.arrays.append([])

            # if current slot is empty, place carry here
            if not self.arrays[i]:
                self.arrays[i] = carry
                break

            # merge existing array with carry
            carry = self._merge(self.arrays[i], carry)
            self.arrays[i] = []
            i += 1

        self.n += 1

    def _binary_search(self, arr, x):
        """
        Perform binary search on a sorted array.
        Runs in O(log len(arr)).
        """
        left = 0
        right = len(arr) - 1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid] == x:
                return True
            elif arr[mid] < x:
                left = mid + 1
            else:
                right = mid - 1

        return False

    def _merge(self, a, b):
        """
        Merge two sorted arrays of equal size.
        Runs in O(len(a) + len(b)).
        """
        merged = []
        i = j = 0

        while i < len(a) and j < len(b):
            if a[i] <= b[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1

        merged.extend(a[i:])
        merged.extend(b[j:])
        return merged

    def __repr__(self):
        return f"BinarySearchableSet(arrays={self.arrays})"


# Example usage
if __name__ == "__main__":
    s = BinarySearchableSet()

    for value in [7, 3, 10, 1, 5, 8, 12]:
        s.insert(value)
        print(s)

    print()
    print(s.search(5))  # True
    print(s.search(9))  # False
