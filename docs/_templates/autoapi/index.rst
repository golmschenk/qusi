
.. toctree::
   :caption: Code Reference
   :titlesonly:

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}

This page contains auto-generated API reference documentation
`Sphinx-AutoAPI <https://github.com/rtfd/sphinx-autoapi>`_.
