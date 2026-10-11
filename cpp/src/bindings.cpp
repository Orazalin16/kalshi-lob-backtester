// Python bindings. The tape is passed as a NumPy structured array and read in
// place; nothing is copied on the way in.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <span>
#include <stdexcept>

#include "klob/backtester.hpp"

namespace py = pybind11;
using namespace klob;

namespace {

using EventArray = py::array_t<Event, py::array::c_style>;

std::span<const Event> as_span(const EventArray& a) {
  if (a.ndim() != 1) throw std::invalid_argument("events must be a 1-D array");
  return {a.data(), static_cast<std::size_t>(a.size())};
}

Side to_side(int s) {
  if (s != 0 && s != 1) throw std::invalid_argument("side must be 0 (YES) or 1 (NO)");
  return static_cast<Side>(s);
}

// Owns the array so the tape outlives the backtester that points into it.
struct PyBacktester {
  PyBacktester(EventArray events, const SimConfig& cfg)
      : tape(std::move(events)), bt(as_span(tape), cfg) {}
  EventArray tape;
  Backtester bt;
};

py::tuple depth_arrays(const Book& book, int side, std::size_t k) {
  py::array_t<Price> px(static_cast<py::ssize_t>(k));
  py::array_t<Qty> sz(static_cast<py::ssize_t>(k));
  const std::size_t n = book.depth(to_side(side), k, px.mutable_data(), sz.mutable_data());
  px.resize({static_cast<py::ssize_t>(n)});
  sz.resize({static_cast<py::ssize_t>(n)});
  return py::make_tuple(px, sz);
}

py::dict stats_dict(const BookStats& s) {
  py::dict d;
  d["deltas"] = s.deltas;
  d["trades"] = s.trades;
  d["snapshots"] = s.snapshots;
  d["gaps"] = s.gaps;
  d["negative_clamps"] = s.negative_clamps;
  d["bad_events"] = s.bad_events;
  return d;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core: L2 order book, queue-position fill simulator, tape replay";

  PYBIND11_NUMPY_DTYPE(Event, ts, exch_ts, qty, price, type, side, flags);
  PYBIND11_NUMPY_DTYPE(Fill, ts, order_id, qty, fee, price, side);
  m.attr("EVENT_DTYPE") = py::dtype::of<Event>();
  m.attr("FILL_DTYPE") = py::dtype::of<Fill>();
  m.attr("PRICE_MAX") = kPriceMax;

  py::enum_<QueueModel>(m, "QueueModel")
      .value("CONSERVATIVE", QueueModel::Conservative)
      .value("PROPORTIONAL", QueueModel::Proportional)
      .value("OPTIMISTIC", QueueModel::Optimistic);

  py::enum_<OrderState>(m, "OrderState")
      .value("PENDING_NEW", OrderState::PendingNew)
      .value("ACTIVE", OrderState::Active)
      .value("FILLED", OrderState::Filled)
      .value("CANCELED", OrderState::Canceled)
      .value("REJECTED", OrderState::Rejected);

  py::class_<SimConfig>(m, "SimConfig")
      .def(py::init([](QueueModel queue_model, TimeNs order_latency_ns, TimeNs cancel_latency_ns,
                       TimeNs attribution_window_ns, std::int64_t maker_fee_ppm,
                       Micro balance_precision_micro) {
             SimConfig c;
             c.queue_model = queue_model;
             c.order_latency = order_latency_ns;
             c.cancel_latency = cancel_latency_ns;
             c.attribution_window = attribution_window_ns;
             c.fees.maker_ppm = maker_fee_ppm;
             c.fees.balance_precision = balance_precision_micro;
             return c;
           }),
           py::arg("queue_model") = QueueModel::Proportional, py::arg("order_latency_ns") = 0,
           py::arg("cancel_latency_ns") = 0, py::arg("attribution_window_ns") = 50'000'000,
           py::arg("maker_fee_ppm") = 0, py::arg("balance_precision_micro") = 10'000)
      .def_readwrite("queue_model", &SimConfig::queue_model)
      .def_readwrite("order_latency_ns", &SimConfig::order_latency)
      .def_readwrite("cancel_latency_ns", &SimConfig::cancel_latency)
      .def_readwrite("attribution_window_ns", &SimConfig::attribution_window);

  py::class_<Order>(m, "Order")
      .def_readonly("id", &Order::id)
      .def_readonly("submit_ts", &Order::submit_ts)
      .def_readonly("active_ts", &Order::active_ts)
      .def_readonly("qty", &Order::qty)
      .def_readonly("remaining", &Order::remaining)
      .def_readonly("ahead", &Order::ahead)
      .def_readonly("price", &Order::price)
      .def_property_readonly("side", [](const Order& o) { return static_cast<int>(o.side); })
      .def_readonly("state", &Order::state)
      .def_property_readonly("cancel_requested", [](const Order& o) { return o.cancel_ts != kNever; });

  py::class_<Book>(m, "Book")
      .def(py::init<>())
      .def("apply", [](Book& b, const EventArray& ev) {
        const auto span = as_span(ev);
        py::gil_scoped_release nogil;
        return replay(b, span);
      }, "Applies a tape to the book and returns the number of events.")
      .def("qty", [](const Book& b, int side, Price p) {
        if (!Book::in_range(p)) throw std::out_of_range("price outside [0, 10000]");
        return b.qty(to_side(side), p);
      })
      .def("best", [](const Book& b, int side) { return b.best(to_side(side)); })
      .def("depth", [](const Book& b, int side, std::size_t k) { return depth_arrays(b, side, k); },
           py::arg("side"), py::arg("k") = 10)
      .def_property_readonly("yes_bid", &Book::yes_bid)
      .def_property_readonly("yes_ask", &Book::yes_ask)
      .def_property_readonly("valid", &Book::valid)
      .def_property_readonly("crossed", &Book::crossed)
      .def_property_readonly("stats", [](const Book& b) { return stats_dict(b.stats()); });

  py::class_<PyBacktester>(m, "Backtester")
      .def(py::init<EventArray, const SimConfig&>(), py::arg("events").noconvert(),
           py::arg("config") = SimConfig{})
      .def("run_until", [](PyBacktester& s, TimeNs t) { return s.bt.run_until(t); })
      .def("elapse", [](PyBacktester& s, TimeNs dt) { return s.bt.elapse(dt); })
      .def("run_all", [](PyBacktester& s) {
        py::gil_scoped_release nogil;
        s.bt.run_until(kNever - 1);
      }, "Replays the rest of the tape without returning to Python.")
      .def("submit", [](PyBacktester& s, int side, Price price, Qty qty) {
        return s.bt.submit(to_side(side), price, qty);
      }, py::arg("side"), py::arg("price"), py::arg("qty"))
      .def("cancel", [](PyBacktester& s, std::uint64_t id) { return s.bt.cancel(id); })
      .def("qty", [](const PyBacktester& s, int side, Price p) {
        if (!Book::in_range(p)) throw std::out_of_range("price outside [0, 10000]");
        return s.bt.book().qty(to_side(side), p);
      })
      .def("best", [](const PyBacktester& s, int side) { return s.bt.book().best(to_side(side)); })
      .def("depth", [](const PyBacktester& s, int side, std::size_t k) {
        return depth_arrays(s.bt.book(), side, k);
      }, py::arg("side"), py::arg("k") = 10)
      .def("top", [](const PyBacktester& s) {
        const Book& b = s.bt.book();
        const Price yb = b.best(Side::Yes);
        const Price nb = b.best(Side::No);
        return py::make_tuple(yb, yb < 0 ? 0 : b.qty(Side::Yes, yb), nb, nb < 0 ? 0 : b.qty(Side::No, nb));
      }, "(best YES bid, its size, best NO bid, its size); a price of -1 means that side is empty.")
      .def("orders", [](const PyBacktester& s) { return s.bt.sim().live_orders(); })
      .def("drain_closed", [](PyBacktester& s) {
        auto out = s.bt.sim().closed_orders();
        s.bt.sim().clear_closed();
        return out;
      })
      .def("fills", [](const PyBacktester& s) {
        const auto& f = s.bt.sim().fills();
        py::array_t<Fill> out(static_cast<py::ssize_t>(f.size()));
        std::copy(f.begin(), f.end(), out.mutable_data());
        return out;
      })
      .def_property_readonly("n_fills", [](const PyBacktester& s) { return s.bt.sim().fills().size(); })
      .def_property_readonly("now", [](const PyBacktester& s) { return s.bt.now(); })
      .def_property_readonly("done", [](const PyBacktester& s) { return s.bt.done(); })
      .def_property_readonly("cursor", [](const PyBacktester& s) { return s.bt.cursor(); })
      .def_property_readonly("valid", [](const PyBacktester& s) { return s.bt.book().valid(); })
      .def_property_readonly("position", [](const PyBacktester& s) { return s.bt.sim().account().position; })
      .def_property_readonly("cash", [](const PyBacktester& s) { return s.bt.sim().account().cash; })
      .def_property_readonly("fees", [](const PyBacktester& s) { return s.bt.sim().account().fees; })
      .def_property_readonly("volume", [](const PyBacktester& s) { return s.bt.sim().account().volume; })
      .def_property_readonly("last_trade", [](const PyBacktester& s) { return s.bt.last_trade_yes_price(); })
      .def_property_readonly("book_stats", [](const PyBacktester& s) { return stats_dict(s.bt.book().stats()); })
      .def_property_readonly("sim_stats", [](const PyBacktester& s) {
        const SimStats& st = s.bt.sim().stats();
        py::dict d;
        d["submitted"] = st.submitted;
        d["rejected"] = st.rejected;
        d["canceled"] = st.canceled;
        d["filled"] = st.filled;
        d["cancel_qty"] = st.cancel_qty;
        d["matched_trade_qty"] = st.matched_trade_qty;
        d["unmatched_trade_qty"] = st.unmatched_trade_qty;
        return d;
      });
}
