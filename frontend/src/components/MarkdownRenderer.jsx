import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * Renders Markdown content properly with GitHub-flavored markdown support.
 * Converts markdown headers, bold, lists, and tables into styled React elements
 * without displaying raw syntax (e.g. '###', '**') to users.
 */
export default function MarkdownRenderer({ content = '', className = '' }) {
  if (!content) return null;

  return (
    <div className={`markdown-body-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ node, ...props }) => (
            <div className="table-responsive-wrapper">
              <table className="comparison-markdown-table" {...props} />
            </div>
          ),
          thead: ({ node, ...props }) => <thead className="table-thead" {...props} />,
          tbody: ({ node, ...props }) => <tbody className="table-tbody" {...props} />,
          tr: ({ node, ...props }) => <tr className="table-row" {...props} />,
          th: ({ node, ...props }) => <th className="table-header-cell" {...props} />,
          td: ({ node, ...props }) => <td className="table-data-cell" {...props} />,
          h1: ({ node, ...props }) => <h3 className="md-header md-h1" {...props} />,
          h2: ({ node, ...props }) => <h3 className="md-header md-h2" {...props} />,
          h3: ({ node, ...props }) => <h4 className="md-header md-h3" {...props} />,
          h4: ({ node, ...props }) => <h5 className="md-header md-h4" {...props} />,
          p: ({ node, ...props }) => <p className="md-para" {...props} />,
          ul: ({ node, ...props }) => <ul className="md-unordered-list" {...props} />,
          ol: ({ node, ...props }) => <ol className="md-ordered-list" {...props} />,
          li: ({ node, ...props }) => <li className="md-list-item" {...props} />,
          strong: ({ node, ...props }) => <strong className="md-bold" {...props} />,
          em: ({ node, ...props }) => <em className="md-italic" {...props} />,
          code: ({ node, inline, ...props }) => (
            <code className={inline ? 'md-code-inline' : 'md-code-block'} {...props} />
          ),
          blockquote: ({ node, ...props }) => <blockquote className="md-blockquote" {...props} />,
          hr: ({ node, ...props }) => <hr className="md-divider" {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
