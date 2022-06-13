from interval import interval


class Interval(interval):

    def format(self, fs):
        """Format into a string using fs as format for the interval bounds.

        The argument fs can be any string format valid with floats:

            >>> interval[-2.1, 3.4].format("%+g")
            [-2.1, +3.4]

        """
        return ', '.join('[' + ', '.join(fs % x for x in sorted(set(c))) + ']' for c in self)
