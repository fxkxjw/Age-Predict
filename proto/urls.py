from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^blog/', include('blog.urls')),

    url(r'^$', 'tmpl.views.facePredict'),
    #url(r'^tmpl/', 'tmpl.views.tmpl', name = 'template'),
    url(r'^facePredict/', 'tmpl.views.testUpload'),
    #url(r'^phone/','tmpl.views.phoneFacePredict'),
    url(r'^imageUpload', 'tmpl.views.imageUpload', name = 'imageUpload'),

    url(r'^images/', 'tmpl.views.imageClassify', name = 'image_classify'),
    url(r'^Classify','tmpl.views.Classify'),

    url(r'^source/(?P<path>.*)', 'django.views.static.serve', {'document_root':'./source/'}),
    url(r'^Images/(?P<path>.*)', 'django.views.static.serve', {'document_root':'./tmpl/images/'}),

    url(r'^admin/', include(admin.site.urls)),
)


