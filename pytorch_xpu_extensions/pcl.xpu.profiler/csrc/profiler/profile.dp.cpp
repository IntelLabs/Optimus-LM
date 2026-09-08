#include <torch/extension.h>

#include <ATen/xpu/XPUContext.h>

#include <sycl/sycl.hpp>
namespace syclex = sycl::ext::oneapi::experimental;

class EventWrapper {
    sycl::event e;
    public:
        EventWrapper(sycl::event e){
            this->e = e;
        }
        unsigned long get_submit_time(){
            return e.get_profiling_info<sycl::info::event_profiling::command_submit>();
        }

        unsigned long get_start_time(){
            return e.get_profiling_info<sycl::info::event_profiling::command_start>();
        }

        unsigned long get_end_time(){
            return e.get_profiling_info<sycl::info::event_profiling::command_end>();
        }

        unsigned long get_elapsed_time(){
            return (e.get_profiling_info<sycl::info::event_profiling::command_end>() - 
                    e.get_profiling_info<sycl::info::event_profiling::command_start>());
        }

        void wait(){
            e.wait();
        }
};

EventWrapper mark_event(){
    auto q = c10::xpu::getCurrentXPUStream().queue();

    sycl::event e = syclex::submit_profiling_tag(q);
    EventWrapper ew(e);
    return ew;
}

EventWrapper async_memcpy(at::Tensor src, at::Tensor dst){
    auto q = c10::xpu::getCurrentXPUStream().queue();

    void* src_ptr = (void*) src.data_ptr();
    void* dst_ptr = (void*) dst.data_ptr();
    auto numBytes = src.numel() * src.element_size();
    
    sycl::event e = q.memcpy(dst_ptr, src_ptr, numBytes);
    
    EventWrapper ew(e);
    return ew;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // py::class_<EventWrapper>(m, "EventWrapper");
    py::class_<EventWrapper>(m, "EventWrapper")
        .def("get_submit_time", &EventWrapper::get_submit_time)
        .def("get_start_time", &EventWrapper::get_start_time)
        .def("get_end_time", &EventWrapper::get_end_time)
        .def("get_elapsed_time", &EventWrapper::get_elapsed_time)
        .def("wait", &EventWrapper::wait);

    m.def("mark_event", &mark_event, "Runs an empty kernel  and return the event");
    m.def("async_memcpy", &async_memcpy, "Copies data asynchronously");

}