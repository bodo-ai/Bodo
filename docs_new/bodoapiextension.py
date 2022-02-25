from markdown.extensions import Extension
from markdown.inlinepatterns import SimpleTagPattern


FULLAPI_RE = r'(\+\+)(.*?)\+\+'
NAMEAPI_RE = r'(\%\%)(.*?)\%\%'

class bodoapi(Extension):

    def extendMarkdown(self, md):
        # Create the del pattern
        api_tag = SimpleTagPattern(FULLAPI_RE, 'apihead')
        api_name_tag = SimpleTagPattern(NAMEAPI_RE, 'apiname')
        # Insert del pattern into markdown parser
        md.inlinePatterns.add('apihead', api_tag, '>not_strong')
        md.inlinePatterns.add('apiname', api_name_tag, '>not_strong')
