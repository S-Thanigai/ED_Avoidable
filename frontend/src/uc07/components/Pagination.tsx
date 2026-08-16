import "./Pagination.css";

export const PAGE_SIZE = 25;

interface PaginationProps {
  page: number;
  totalItems: number;
  onPageChange: (page: number) => void;
}

/** Purely presentational pagination -- the caller owns slicing the data;
 * this component only renders controls and reports the requested page. */
export function Pagination({ page, totalItems, onPageChange }: PaginationProps) {
  const pageCount = Math.max(1, Math.ceil(totalItems / PAGE_SIZE));
  const clampedPage = Math.min(Math.max(1, page), pageCount);
  const start = totalItems === 0 ? 0 : (clampedPage - 1) * PAGE_SIZE + 1;
  const end = Math.min(clampedPage * PAGE_SIZE, totalItems);

  if (totalItems === 0) return null;

  // Keep the page-number list compact: always show first, last, current,
  // and one neighbor on each side; collapse the rest with an ellipsis.
  const pageNumbers: (number | "ellipsis")[] = [];
  for (let p = 1; p <= pageCount; p++) {
    if (p === 1 || p === pageCount || Math.abs(p - clampedPage) <= 1) {
      pageNumbers.push(p);
    } else if (pageNumbers[pageNumbers.length - 1] !== "ellipsis") {
      pageNumbers.push("ellipsis");
    }
  }

  return (
    <nav className="pagination" aria-label="Member table pagination">
      <span className="pagination__summary">
        Showing {start}–{end} of {totalItems} members
      </span>
      <div className="pagination__controls">
        <button
          type="button"
          className="pagination__nav"
          disabled={clampedPage <= 1}
          onClick={() => onPageChange(clampedPage - 1)}
        >
          Previous
        </button>
        {pageNumbers.map((p, idx) =>
          p === "ellipsis" ? (
            <span key={`ellipsis-${idx}`} className="pagination__ellipsis" aria-hidden="true">
              …
            </span>
          ) : (
            <button
              key={p}
              type="button"
              className={`pagination__page${p === clampedPage ? " pagination__page--active" : ""}`}
              aria-current={p === clampedPage ? "page" : undefined}
              onClick={() => onPageChange(p)}
            >
              {p}
            </button>
          ),
        )}
        <button
          type="button"
          className="pagination__nav"
          disabled={clampedPage >= pageCount}
          onClick={() => onPageChange(clampedPage + 1)}
        >
          Next
        </button>
      </div>
    </nav>
  );
}
